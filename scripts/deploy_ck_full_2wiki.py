"""ck_full end to end against the GNN on one machine, track MP-APPROX (configs/deploy_ck_full_2wiki.yaml): deploy_ck_2wiki's
nine sealed fits (twin, ck_full and gnn, seeds 0-2) scored again on the laptop CPU with the fast feature compiler wired in
and checked against the reference on every query, then timed from a query's cached first-stage lists to its top-5 at a
batch of one, the kernel timed inside the forward.

    python scripts/deploy_ck_full_2wiki.py --stage smoke             # laptop: every stage on a shard and 16 sampled queries (never read)
    python scripts/deploy_ck_full_2wiki.py --stage score             # laptop: the population scored by the nine fits
    python scripts/deploy_ck_full_2wiki.py --stage latency-prepare   # laptop: the sample, the pool and compile checks, the cached top-5
    python scripts/deploy_ck_full_2wiki.py --stage latency           # laptop: the driver (laptop state, 3 rounds x 3 arms, then memory)
    python scripts/deploy_ck_full_2wiki.py --stage read              # laptop: retention and speed bands, flags, the Pareto point
    python scripts/deploy_ck_full_2wiki.py --stage doc               # laptop: record.json and docs/DEPLOY_CK_FULL_2WIKI.md
    python scripts/deploy_ck_full_2wiki.py --stage file --date 2026_10_01 --commit <sha> [--extra run_extra.json]

The driver starts the timed processes (--stage latency-run --arm A --round R) and the memory processes (--stage memory-run
--arm A --round R) itself. ck_full is a compressed, query-conditioned one-hop message-passing approximation: its kernel is
learned propagation by configs/universal_v2.yaml#learned_vs_fixed_propagation. deploy_ck_2wiki.py is imported unchanged;
the four module globals scoring.rebinding lists are rebound in memory in every process of this file, so no hard stop of
the imported code can write into deploy_ck_2wiki's outputs.
"""

from __future__ import annotations

import os
import sys

THREADS = 8
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")

if __name__ == "__main__":   # placement: the BLAS, OpenMP and numba pools are fixed before numpy, numba and torch load
    for _var in BLAS_ENV:
        os.environ[_var] = str(THREADS)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import deploy_ck_2wiki as DK  # noqa: E402  (imported, never edited)
import universal_v2_run as V2  # noqa: E402  (imported, never edited)
from mp_retrieval import fast_features as FF  # noqa: E402  (imported, never edited: its freeze stands)
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import QueryInputs  # noqa: E402
from mp_retrieval.m3b_models import PackedBatch  # noqa: E402
from mp_retrieval.universal_v2_features import COLUMNS, compile_query_v2  # noqa: E402
from mp_retrieval.universal_v2_models import RelationBank, pack_queries_v2  # noqa: E402

CONFIG = ROOT / "configs" / "deploy_ck_full_2wiki.yaml"
OUT = ROOT / "outputs" / "deploy_ck_full_2wiki"
DOC = ROOT / "docs" / "DEPLOY_CK_FULL_2WIKI.md"
DK_FITS = Path("outputs") / "deploy_ck_2wiki" / "fits"                          # relative to the repository root
DK_EVAL_NPZ = Path("outputs") / "deploy_ck_2wiki" / "eval" / "2wiki.npz"
DK_EVAL_IDS = Path("outputs") / "deploy_ck_2wiki" / "eval" / "2wiki_query_ids.json"
REFERENCE = Path("outputs") / "deploy_ck_2wiki_reference" / "batches_laptop.json"
PHASE = "DEPLOY_CK_FULL_2WIKI"
NAME = DK.NAME
ARMS = ("twin", "ck_full", "gnn")
SEEDS = DK.SEEDS
HALVES = ("V2_HELD_CONFIRMATION", "V2_GATE", "whole")        # the primary half first; the whole population descriptive
READ_METRICS = ("recall@5", "full_coverage@5", "hit@1")      # the reading's metric, its co-read, and the descriptive one
READINGS = ("GNN_GAIN_ABSENT", "FULL_KEEPS", "FULL_PARTIAL", "FULL_NO_GAIN")
SPEED_BANDS = ("E2E_FASTER", "E2E_MARGINAL", "E2E_NOT_FASTER")
FLAGS = ("FULL_NOT_BELOW_GNN", "GATE_HALF_DIFFERS", "FC5_DIFFERS", "E2E_SLOWER", "TAIL_DIFFERS", "FORWARD_ONLY_DIFFERS",
         "MEMORY_ABOVE_GNN", "E2E_PATH_DIFFERS", "LAPTOP_NOT_IDLE")
CONTRASTS = {"gnn_gain": ("gnn", "twin"), "full_gain": ("ck_full", "twin"), "full_vs_gnn": ("ck_full", "gnn")}
SHARES = {"full_share": (("ck_full", "twin"), ("gnn", "twin"))}
RATIOS = {"ck_full/gnn": ("ck_full", "gnn"), "twin/gnn": ("twin", "gnn"), "ck_full/twin": ("ck_full", "twin")}
KEEP = {"share": 0.75, "share_low": 0.50}
SPEED = {"faster": 0.90, "one": 1.00}
QS = (50, 95, 99)
ORDERS = (("twin", "ck_full", "gnn"), ("ck_full", "gnn", "twin"), ("gnn", "twin", "ck_full"))   # latency.rounds
PASSES = ("e2e_fast", "e2e_reference", "breakdown")
STAGES = ("pool", "gather", "compile", "pack", "forward", "topk")
NODE_LOCAL = ("input", "body", "readout")                     # the modules every arm shares with the twin
IDLE = {"cpu_percent": 15.0, "sample_seconds": 3.0, "wait_minutes": 20.0, "poll_seconds": 30.0}
AC_WAIT_MINUTES = 30.0
CEILING_HOURS = 4.0
REBOUND = ("compile_eval_query", "EVAL", "OUT", "LATENCY_QUERIES")   # scoring.rebinding, exactly these
SIZES = {False: {"sample": 1024, "warm": 64, "rounds": 3, "top_memory": 128, "shard": None},
         True: {"sample": 16, "warm": 4, "rounds": 1, "top_memory": 8, "shard": (0, 64)}}   # the smoke's (never read)
EMPTY_GOLD = np.zeros(0, dtype=np.int64)
LF = chr(10)

log_utc = DK.log_utc
sha256_file = V2.sha256_file
lf_sha256 = V2.lf_sha256
read_json = DK.read_json
atomic_json = DK.atomic_json
clean = DK.clean


@dataclasses.dataclass(frozen=True)
class Where:
    """Where a run's files go: the declared outputs, or the smoke's beside them (never read, never filed)."""

    out: Path
    doc: Path
    smoke: bool = False

    @property
    def eval(self) -> Path:
        return self.out / "eval"

    @property
    def latency(self) -> Path:
        return self.out / "latency"

    @property
    def memory(self) -> Path:
        return self.out / "memory"

    @property
    def read(self) -> Path:
        return self.out / "read.json"

    @property
    def record(self) -> Path:
        return self.out / "record.json"


MAIN = Where(OUT, DOC)
SMOKE = Where(OUT / "smoke", OUT / "smoke" / "DEPLOY_CK_FULL_2WIKI_smoke.md", smoke=True)


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence is written to outputs/deploy_ck_full_2wiki/hard_stop.json and nothing further runs here."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hard_stop.json").write_text(json.dumps(clean({"utc": V2.utc(), "message": message, **evidence}), indent=1), encoding="utf-8")
    raise SystemExit(f"HARD STOP: {message}")


# ── the declaration and its pins ─────────────────────────────────────────────


def load_declaration(path: Path = CONFIG) -> dict:
    decl = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if decl.get("phase") != PHASE:
        raise SystemExit(f"{path}: not the {PHASE} declaration")
    if tuple(decl["arms"]) != ARMS:
        raise SystemExit(f"arms: {list(decl['arms'])}, this script runs {list(ARMS)}")
    for arm, spec in decl["arms"].items():
        if int(spec["parameters"]) != DK.ARM_SPECS[arm]["parameters"]:
            raise SystemExit(f"arms.{arm}: the parameter pin is not deploy_ck_2wiki's")
        if list(spec["fits"]) != [DK.fit_key(arm, s) for s in SEEDS]:
            raise SystemExit(f"arms.{arm}.fits: not deploy_ck_2wiki's keys for seeds {list(SEEDS)}")
    r = decl["readings"]
    ret = r["retention"]
    if (r["primary_half"], ret["metric"], ret["co_read"], ret["descriptive"]) != (HALVES[0], *READ_METRICS):
        raise SystemExit("readings: the primary half or the metrics are not this script's")
    if list(ret["bands"]) != list(READINGS) or list(r["speed"]["bands"]) != list(SPEED_BANDS) or list(r["flags"]) != list(FLAGS):
        raise SystemExit("readings: the bands or the flags are not this script's")
    st = decl["statistics"]
    if {c: f"{a} - {b}" for c, (a, b) in CONTRASTS.items()} != dict(st["contrasts"]) or list(st["shares"]) != list(SHARES):
        raise SystemExit("statistics: the contrasts or the shares are not this script's")
    lat = decl["latency"]
    if (int(str(lat["sample"]["timed"]).split()[0]) != SIZES[False]["sample"] or int(str(lat["sample"]["warm_up"]).split()[0]) != SIZES[False]["warm"]
            or not str(lat["rounds"]).strip().startswith(f"{SIZES[False]['rounds']} rounds")):
        raise SystemExit("latency: the sample sizes or the rounds are not this script's")
    return decl


def pin_problems(decl: dict, root: Path = ROOT) -> dict:
    """inputs and arms: the frozen code (LF sha256), the pinned files and every fit's weights and record (file bytes)."""
    bad = {}
    for rel, want in decl["inputs"]["frozen_code_lf"].items():
        p = root / rel
        got = lf_sha256(p) if p.exists() else None
        if got != want:
            bad[rel] = {"declared": want, "found": got}
    for rel, want in decl["inputs"]["files_sha256"].items():
        p = root / rel
        got = sha256_file(p) if p.exists() else None
        if got != want:
            bad[rel] = {"declared": want, "found": got}
    for spec in decl["arms"].values():
        for key, pins in spec["fits"].items():
            for suffix, field in ((".pt", "weights_sha256"), (".json", "record_sha256")):
                p = root / DK_FITS / f"{key}{suffix}"
                got = sha256_file(p) if p.exists() else None
                if got != pins[field]:
                    bad[f"{key}{suffix}"] = {"declared": pins[field], "found": got}
    return bad


def verify_pins(decl: dict) -> None:
    bad = pin_problems(decl)
    if bad:
        hard_stop("pinned inputs differ from configs/deploy_ck_full_2wiki.yaml", files=bad)


def load_fits(decl: dict, inputs: dict, bank, keys=None) -> dict[str, torch.nn.Module]:
    """The sealed fits (all nine, or ``keys``), each held to its pins: the weights' bytes and content, the record's own
    sha256 fields; built by deploy_ck_2wiki.build_model (its parameter pins), in eval mode on the CPU."""
    models = {}
    for arm, spec in decl["arms"].items():
        for key, pins in spec["fits"].items():
            if keys is not None and key not in keys:
                continue
            pt, js = ROOT / DK_FITS / f"{key}.pt", ROOT / DK_FITS / f"{key}.json"
            rec = read_json(js)
            if rec is None or not pt.exists():
                hard_stop(f"{key}: no weights or record in {DK_FITS.as_posix()}")
            if sha256_file(pt) != pins["weights_sha256"] or rec["state_sha256"] != pins["weights_sha256"]:
                hard_stop(f"{key}: the weights are not the pinned file")
            state = torch.load(pt, map_location="cpu")
            if DK.state_digest(state) != pins["content_sha256"] or rec["state_content_sha256"] != pins["content_sha256"]:
                hard_stop(f"{key}: the weights' content is not the pinned content")
            model = DK.build_model(arm, inputs, bank)
            model.load_state_dict(state)
            model.eval()
            models[key] = model
    return models


def load_reference(decl: dict, root: Path = ROOT) -> tuple[dict, str]:
    """Amendment 2's laptop reference, the file this declaration pins, packed at this file's thread count."""
    want = decl["inputs"]["files_sha256"][REFERENCE.as_posix()]
    got = sha256_file(root / REFERENCE)
    if got != want:
        hard_stop("the laptop's batch reference is not the pinned file", declared=want, found=got)
    ref = read_json(root / REFERENCE)
    if int(ref["blas_threads"]) != THREADS or int(ref["threads"]) != THREADS:
        hard_stop("the laptop's batch reference was not packed at this file's thread count", blas_threads=ref["blas_threads"], threads=ref["threads"])
    return ref, got


# ── scoring.rebinding and the checked compile ─────────────────────────────────


@contextlib.contextmanager
def rebound(compile_fn, eval_dir: Path, out_dir: Path | None = None):
    """scoring.rebinding: deploy_ck_2wiki's compile_eval_query, EVAL, OUT and LATENCY_QUERIES rebound in this process,
    restored on exit; nothing else of the imported module changes and no file is edited. OUT is this file's own."""
    saved = {k: getattr(DK, k) for k in REBOUND}
    DK.compile_eval_query, DK.EVAL, DK.OUT, DK.LATENCY_QUERIES = compile_fn, Path(eval_dir), Path(out_dir or OUT), 0
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(DK, k, v)


def _bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.uint32) if a.dtype == np.float32 else a


def compiled_problems(fast, ref) -> dict:
    """tests/test_fast_features.py's comparison as a list of differences: the pool, the seeds, the scalars and the seed
    weights bit for bit, and every family's edge arrays (float32 by their bits)."""
    bad = {}
    if fast.pool.dtype != ref.pool.dtype or not np.array_equal(fast.pool, ref.pool):
        bad["pool"] = {"fast": [str(fast.pool.dtype), int(fast.pool.size)], "reference": [str(ref.pool.dtype), int(ref.pool.size)]}
    if not np.array_equal(fast.seeds_local, ref.seeds_local):
        bad["seeds_local"] = {"fast": np.asarray(fast.seeds_local).tolist(), "reference": np.asarray(ref.seeds_local).tolist()}
    if fast.scalars.shape != ref.scalars.shape or fast.scalars.dtype != ref.scalars.dtype:
        bad["scalars"] = {"fast": [list(fast.scalars.shape), str(fast.scalars.dtype)], "reference": [list(ref.scalars.shape), str(ref.scalars.dtype)]}
    else:
        diff = _bits(fast.scalars) != _bits(ref.scalars)
        cols = np.flatnonzero(diff.any(axis=0))
        if cols.size:
            bad["scalars"] = {"columns": [COLUMNS[j] if j < len(COLUMNS) else int(j) for j in cols], "cells": int(diff.sum())}
    if (fast.seedw.dtype != ref.seedw.dtype or fast.seedw.shape != ref.seedw.shape
            or not np.array_equal(_bits(fast.seedw), _bits(ref.seedw))):
        bad["seedw"] = {"fast": [list(fast.seedw.shape), str(fast.seedw.dtype)], "reference": [list(ref.seedw.shape), str(ref.seedw.dtype)]}
    if set(fast.edges) != set(ref.edges):
        bad["edge_families"] = {"fast": sorted(fast.edges), "reference": sorted(ref.edges)}
    else:
        for fam in ref.edges:
            fa, ra = fast.edges[fam], ref.edges[fam]
            if len(fa) != len(ra):
                bad[f"edges/{fam}"] = {"fast_arrays": len(fa), "reference_arrays": len(ra)}
                continue
            for j, (a, b) in enumerate(zip(fa, ra)):
                if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(_bits(a), _bits(b)):
                    bad[f"edges/{fam}/{j}"] = {"fast": [str(a.dtype), list(a.shape)], "reference": [str(b.dtype), list(b.shape)]}
    return bad


class CheckedCompile:
    """scoring.rebinding.compile_eval_query: the dense rows gathered as before, the fast compiler at 8 threads, and the
    reference compile_query_v2 on the same inputs; a difference in any array is a hard stop; the fast compile is returned."""

    def __init__(self, threads: int = THREADS):
        self.threads = threads
        self.queries = 0
        self.fast_seconds = 0.0
        self.reference_seconds = 0.0

    def compiler(self, context):
        fc = FF.compiler_for(context.stores, context.nodes, context.rel_table, threads=self.threads)
        if fc is None:
            hard_stop("the fast compiler does not serve 2wiki's context (numba is missing or the context is unsupported)")
        return fc

    def check(self, inp, pool, seeds, context, E, i: int):
        fc = self.compiler(context)
        t0 = time.perf_counter()
        fast = fc.compile(inp, pool, seeds, embeddings=E, v2=True)
        t1 = time.perf_counter()
        ref = compile_query_v2(inp, pool, seeds, context.stores, context.nodes, context.rel_table, embeddings=E)
        t2 = time.perf_counter()
        bad = compiled_problems(fast, ref)
        if bad:
            hard_stop(f"query {int(i)}: the fast compile differs from the reference compile_query_v2", query=int(i), problems=bad)
        self.queries += 1
        self.fast_seconds += t1 - t0
        self.reference_seconds += t2 - t1
        return fast

    def __call__(self, prep, context, i: int):
        E = context.nodes.read(prep.pools[i])
        inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
        return E, self.check(inp, prep.pools[i], prep.seeds[i], context, E, int(i))

    def summary(self) -> dict:
        return {"queries_checked": self.queries, "equal_on_every_query": True, "threads": self.threads,
                "fast_seconds": round(self.fast_seconds, 2), "reference_seconds": round(self.reference_seconds, 2)}


def laptop_placement() -> dict:
    """placement.machine: the thread pools read back (a difference is a hard stop) and the versions filed."""
    import platform

    env = {v: os.environ.get(v) for v in BLAS_ENV}
    if any(x != str(THREADS) for x in env.values()) or torch.get_num_threads() != THREADS:
        hard_stop("placement: the thread pools are not the declared 8", env=env, torch_threads=torch.get_num_threads())
    return {"host": platform.node(), "device": "cpu", "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "numba": getattr(FF.numba, "__version__", None) if FF.numba is not None else None,
            "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(), "blas_env": env, "processor": platform.processor()}


# ── stage: score ─────────────────────────────────────────────────────────────


def host_agreement(arrays: dict, ids: list[str], root: Path = ROOT) -> dict:
    """scoring.host_agreement (systems only): per fit, the queries where a read metric differs from deploy_ck_2wiki's
    host eval arrays, aligned by query id. Filed, never read."""
    with np.load(root / DK_EVAL_NPZ) as z:
        host = {k: z[k] for k in z.files}
    host_ids = json.loads((root / DK_EVAL_IDS).read_text(encoding="utf-8"))
    pos = {q: j for j, q in enumerate(host_ids)}
    where = np.asarray([pos[q] for q in ids], dtype=np.int64)
    out = {}
    for arm in ARMS:
        for s in SEEDS:
            key = DK.fit_key(arm, s)
            differ = np.zeros(len(ids), dtype=bool)
            for m in READ_METRICS:
                differ |= arrays[f"{key}/{m}"] != host[f"{key}/{m}"][where]
            out[key] = {"queries_where_a_metric_differs": int(differ.sum()), "queries": len(ids)}
    return out


def stage_score(decl: dict, where: Where = MAIN, log=log_utc) -> dict:
    """The retention of every fit on the laptop CPU: deploy_ck_2wiki.eval_placed unchanged under the rebinding, with
    amendment 2's batch check (the smoke's shard compares no batches), then the record's file field rewritten."""
    path = where.eval / f"{NAME}.json"
    if path.exists():
        rec = read_json(path)
        if "rebinding" not in rec:   # deploy_ck_2wiki's writer ran, this file's fields did not: never read as finished
            raise SystemExit(f"{DK.shown(path)} lacks this file's fields (the process ended between the two writes); move it aside first")
        log("eval record exists, not repeated")
        return rec
    place = laptop_placement()
    verify_pins(decl)
    shard = SIZES[where.smoke]["shard"]
    checked = CheckedCompile()
    with rebound(checked, where.eval):
        cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = DK.open_2wiki(decl, host=False, log=log)
        ref, ref_sha = load_reference(decl)
        m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
        models = load_fits(decl, inputs, bank)
        log(f"== score: {len(models)} fits on the laptop CPU at {torch.get_num_threads()} threads, the fast compile checked on every query")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rec = DK.eval_placed(decl, cfg, cfg_m3b, cfg_h, inputs, models, {}, contexts[NAME], handles[NAME], pkg, m3b_compile, m3b_contract,
                                 "cpu", shard=shard, check=None if shard else ref["eval"]["batches"], log=log)
    with np.load(where.eval / f"{NAME}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    ids = json.loads((where.eval / f"{NAME}_query_ids.json").read_text(encoding="utf-8"))
    rec["file"] = CONFIG.relative_to(ROOT).as_posix()
    rec["rebinding"] = {"compile_eval_query": "deploy_ck_full_2wiki.CheckedCompile (fast_features.compiler_for at 8 threads, "
                                              "compared with compile_query_v2 on every query)",
                        "EVAL": DK.shown(where.eval), "OUT": DK.shown(OUT), "LATENCY_QUERIES": 0}
    rec["fast_compile"] = checked.summary()
    if rec.get("batch_check") is not None:
        rec["batch_check"]["reference_sha256"] = ref_sha
    rec["host_agreement"] = host_agreement(arrays, ids)
    rec.update({"placement": place, "warnings": DK.warning_summary(caught), "git_head": DK.git_head(), "module_sha256": DK.module_shas()})
    atomic_json(path, rec)
    log(f"score: {checked.queries} queries, fast == reference on every one; eval record {DK.shown(path)}")
    return rec


# ── latency: the sample, the path, the processes ─────────────────────────────


def sample_positions(half: np.ndarray, n: int, gate: bool) -> np.ndarray:
    """latency.sample: ``n`` positions of one half (V2_GATE where ``half`` is True), evenly spaced over its positions in
    population order (numpy linspace over them, rounded); all distinct."""
    pos = np.flatnonzero(half if gate else ~half)
    if n > pos.size:
        raise ValueError(f"{n} positions asked of a half of {pos.size}")
    out = pos[np.rint(np.linspace(0, pos.size - 1, n)).astype(np.int64)]
    if np.unique(out).size != n:
        raise ValueError("the evenly spaced positions are not distinct")
    return out


def top_memory_positions(positions: np.ndarray, sizes: np.ndarray, n: int) -> np.ndarray:
    """latency.memory: the ``n`` sampled positions with the largest pools (ties by position)."""
    order = np.lexsort((positions, -sizes[positions]))
    return positions[order[:n]]


class PoolBuilder:
    """latency.path.pool: the query's pool rebuilt from its cached first-stage lists under the frozen construction, with
    m3b_pools.seeds_of, m3b_compile.base_rows on the query's own row, m3b_pools.expand_hops and m3b_compile.build_pool,
    as m3b_compile.prepare builds it."""

    def __init__(self, construction: dict, cfg_h: dict, stores: dict, m3a, m3b_contract, m3b_compile):
        self.construction, self.m3a, self.m3b_contract, self.m3b_compile = construction, m3a, m3b_contract, m3b_compile
        self.constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
        regime = construction["regime"]
        self.fams = [stores[f] for f in m3b_compile.regime_families(cfg_h, regime)] if regime != "RETRIEVAL" else None
        self.setting = construction.get("setting")

    def __call__(self, d_ids: np.ndarray, s_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seeds = m3b_pools.seeds_of(d_ids, s_ids)
        base = self.m3b_compile.base_rows(self.construction, d_ids[None, :], s_ids[None, :], self.m3a, self.m3b_contract, self.constant)[0]
        expansion = m3b_pools.expand_hops(seeds, self.fams, self.setting) if self.fams is not None else None
        pool, _ = self.m3b_compile.build_pool(np.asarray(base, dtype=np.int64), seeds, expansion)
        return pool, seeds


def serve_entry(comp, E, qemb, columns) -> dict:
    """One compiled query as pack_queries_v2 reads it at serving time: deploy_ck_2wiki.eval_entry's fields without a gold."""
    return {"pool": comp.pool, "x": comp.scalars[:, columns], "seedw": comp.seedw, "qemb": qemb, "seeds": comp.seeds_local,
            "gold": EMPTY_GOLD, "gold_total": 0, "emb": E}


def run_path(q: dict, builder: PoolBuilder, context, fc, model, inputs: dict, columns, reference: bool = False):
    """latency.path for one query at a batch of one: pool -> gather -> compile -> pack -> forward -> top-5, each stage timed
    with time.perf_counter. Returns the stage seconds, the top-5 global ids and the scores."""
    clock = time.perf_counter
    t0 = clock()
    pool, seeds = builder(q["d_ids"], q["s_ids"])
    t1 = clock()
    E = context.nodes.read(pool)
    t2 = clock()
    inp = QueryInputs(q["qemb"], q["d_ids"], q["d_sc"], q["s_ids"], q["s_sc"])
    if reference:
        comp = compile_query_v2(inp, pool, seeds, context.stores, context.nodes, context.rel_table, embeddings=E)
    else:
        comp = fc.compile(inp, pool, seeds, embeddings=E, v2=True)
    t3 = clock()
    batch = pack_queries_v2([serve_entry(comp, E, q["qemb"], columns)], context)
    t4 = clock()
    with torch.no_grad():
        scores = model(V2.arm_view(model, batch, inputs))
    t5 = clock()
    top = torch.topk(scores, min(5, int(scores.shape[0]))).indices.numpy()
    ids = comp.pool[top]
    t6 = clock()
    return {"pool": t1 - t0, "gather": t2 - t1, "compile": t3 - t2, "pack": t4 - t3, "forward": t5 - t4, "topk": t6 - t5, "total": t6 - t0}, ids, scores


class ModuleClock:
    """latency.passes.breakdown: forward pre- and post-hooks on the node-local modules an arm shares with the twin (input,
    body where present, readout) and on ck_full's kernel module; seconds per module for the query in progress."""

    def __init__(self, model: torch.nn.Module):
        self.names = [n for n in (*NODE_LOCAL, "kernel") if isinstance(getattr(model, n, None), torch.nn.Module)]
        self.spent = {n: 0.0 for n in self.names}
        self._start: dict = {}
        self._handles = []
        for n in self.names:
            m = getattr(model, n)
            self._handles.append(m.register_forward_pre_hook(self._pre(n)))
            self._handles.append(m.register_forward_hook(self._post(n)))

    def _pre(self, n):
        def hook(module, args):
            self._start[n] = time.perf_counter()
        return hook

    def _post(self, n):
        def hook(module, args, output):
            self.spent[n] += time.perf_counter() - self._start[n]
        return hook

    def reset(self) -> None:
        for n in self.spent:
            self.spent[n] = 0.0

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def first_stage_rows(prep, positions: np.ndarray) -> dict:
    """The not-charged first-stage lists of the sampled queries, as m3b_compile.prepare sliced them (dtypes kept)."""
    return {"positions": positions.astype(np.int64), "d_ids": np.stack([prep.dense_ids[i] for i in positions]),
            "d_sc": np.stack([prep.dense_scores[i] for i in positions]), "s_ids": np.stack([prep.splade_ids[i] for i in positions]),
            "s_sc": np.stack([prep.splade_scores[i] for i in positions]), "qemb": np.stack([prep.qemb[i] for i in positions])}


def query_row(rows: dict, j: int) -> dict:
    return {k: rows[k][j] for k in ("d_ids", "d_sc", "s_ids", "s_sc", "qemb")}


def batch_fields(batch: PackedBatch) -> dict:
    return {f.name: getattr(batch, f.name) for f in dataclasses.fields(PackedBatch)}


def touch(t: torch.Tensor) -> int:
    """Read every byte of a (memory-mapped) tensor without a full-size copy: the byte sum."""
    a = t.contiguous().numpy()
    return int(a.reshape(-1).view(np.uint8).sum(dtype=np.uint64)) if a.size else 0


def process_memory() -> dict:
    import psutil

    info = psutil.Process().memory_info()
    return {"wset": int(getattr(info, "wset", info.rss)), "peak_wset": int(getattr(info, "peak_wset", info.rss))}


def cpu_frequency() -> dict | None:
    import psutil

    try:
        f = psutil.cpu_freq()
        return None if f is None else {"current_mhz": float(f.current), "max_mhz": float(f.max)}
    except Exception:
        return None


def stage_latency_prepare(decl: dict, where: Where = MAIN, log=log_utc) -> dict:
    """latency.checks_outside_the_clock, before any timing, in one process: the sample; for every sampled and warm-up query
    the rebuilt pool equals the prepared pool and the fast compile equals the reference; each arm's seed-0 scores and top-5
    from the cached compile; the pre-packed batches of the largest pools (memory); the sampled rows' first-stage lists."""
    lat = where.latency
    if (lat / "sample.json").exists():
        log("sample.json exists, not repeated")
        return read_json(lat / "sample.json")
    place = laptop_placement()
    verify_pins(decl)
    size = SIZES[where.smoke]
    checked = CheckedCompile()
    t0 = time.time()
    with rebound(checked, where.eval):
        cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = DK.open_2wiki(decl, host=False, log=log)
        context, ds = contexts[NAME], handles[NAME]
        m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
        ev = DK.eval_population(decl, cfg, cfg_m3b, cfg_h, context, ds, pkg, m3b_compile, m3b_contract, log=log)
        prep, pop, half = ev["prep"], ev["pop"], ev["half"]
        timed = sample_positions(half, size["sample"], gate=False)
        warm = sample_positions(half, size["warm"], gate=True)
        order = np.concatenate([warm, timed])
        builder = PoolBuilder(ev["construction"], cfg_h, context.stores, pkg[0], m3b_contract, m3b_compile)
        bad = [int(i) for i in order if not all(np.array_equal(a, b) for a, b in zip(builder(prep.dense_ids[i], prep.splade_ids[i]),
                                                                                      (prep.pools[i], prep.seeds[i])))]
        if bad:
            hard_stop("a rebuilt pool differs from the prepared pool", queries=bad[:20], count=len(bad))
        sizes = ev["sizes"]
        top_mem = top_memory_positions(timed, sizes, size["top_memory"])
        models = load_fits(decl, inputs, bank, keys={DK.fit_key(a, 0) for a in ARMS})
        columns = inputs["column_indices"]
        cached = {a: {"top5": [], "scores": []} for a in ARMS}
        packed, keep = {}, set(top_mem.tolist())
        for i in order:
            E, comp = checked(prep, context, int(i))   # a difference from the reference is a hard stop
            batch = pack_queries_v2([serve_entry(comp, E, prep.qemb[i], columns)], context)
            if int(i) in keep:
                packed[int(i)] = batch_fields(batch)
            for a in ARMS:
                model = models[DK.fit_key(a, 0)]
                with torch.no_grad():
                    s = model(V2.arm_view(model, batch, inputs))
                top = torch.topk(s, min(5, int(s.shape[0]))).indices.numpy()
                cached[a]["top5"].append(comp.pool[top].tolist())
                cached[a]["scores"].append(s.numpy().tolist())
    lat.mkdir(parents=True, exist_ok=True)
    rows = first_stage_rows(prep, order)
    np.savez(lat / "first_stage.npz", **rows)
    torch.save({"positions": [int(i) for i in top_mem], "query_ids": [pop.ids[int(i)] for i in top_mem],
                "batches": [packed[int(i)] for i in top_mem],
                "bank": {"embeddings": bank.embeddings, "offsets": dict(bank.offsets), "sha256": bank.sha256}}, lat / "packed_top128.pt")
    out = {"file": CONFIG.relative_to(ROOT).as_posix(), "utc": V2.utc(), "smoke": where.smoke, "sizes": size,
           "timed": timed.tolist(), "warm": warm.tolist(), "order": order.tolist(),
           "query_ids": {"timed": [pop.ids[int(i)] for i in timed], "warm": [pop.ids[int(i)] for i in warm]},
           "pool_sizes": {"timed": sizes[timed].tolist(), "warm": sizes[warm].tolist()}, "top_memory": top_mem.tolist(),
           "pools_rebuilt_equal": int(order.size), "fast_compile": checked.summary(), "cached": cached,
           "first_stage_sha256": sha256_file(lat / "first_stage.npz"), "packed_sha256": sha256_file(lat / "packed_top128.pt"),
           "seconds": round(time.time() - t0, 1), "placement": place, "git_head": DK.git_head(), "module_sha256": DK.module_shas()}
    atomic_json(lat / "sample.json", out)
    log(f"latency-prepare: {timed.size} timed and {warm.size} warm-up queries; pools rebuilt equal and fast == reference on all; "
        f"{top_mem.size} batches packed for memory")
    return out


def sample_inputs(where: Where) -> tuple[dict, dict]:
    sample = read_json(where.latency / "sample.json")
    if sample is None:
        hard_stop("no latency/sample.json; latency-prepare runs first")
    fs = where.latency / "first_stage.npz"
    if not fs.exists() or sha256_file(fs) != sample["first_stage_sha256"]:
        hard_stop("latency/first_stage.npz is not the file latency-prepare pinned")
    with np.load(fs) as z:
        rows = {k: z[k] for k in z.files}
    if rows["positions"].tolist() != sample["order"]:
        hard_stop("latency/first_stage.npz holds other positions than the sample's order")
    return sample, rows


def stage_latency_run(decl: dict, arm: str, rnd: int, idle: bool, where: Where = MAIN, log=log_utc) -> dict:
    """One timed process: startup (context, weights, the JIT) filed, then e2e_fast, e2e_reference and breakdown, each after
    the warm-up queries; after each pass every timed query's top-5 and scores are compared with the cached pass's."""
    import psutil

    created = psutil.Process().create_time()
    t_main = time.time()
    freq_start = cpu_frequency()
    place = laptop_placement()
    verify_pins(decl)
    sample, rows = sample_inputs(where)
    n_warm = len(sample["warm"])
    at = {pos: j for j, pos in enumerate(sample["order"])}
    timed_j = [at[p] for p in sample["timed"]]
    key = DK.fit_key(arm, 0)
    with rebound(CheckedCompile(), where.eval):
        t0 = time.time()
        cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = DK.open_2wiki(decl, host=False, log=log)
        context = contexts[NAME]
        m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
        _, frozen = m3b_compile.frozen_contract(cfg_m3b)
        builder = PoolBuilder(frozen["per_dataset"][NAME]["construction"], cfg_h, context.stores, pkg[0], m3b_contract, m3b_compile)
        columns = inputs["column_indices"]
        fc = CheckedCompile().compiler(context)
        t_context = time.time() - t0
        t0 = time.time()
        model = load_fits(decl, inputs, bank, keys={key})[key]
        t_weights = time.time() - t0
        t0 = time.time()
        run_path(query_row(rows, 0), builder, context, fc, model, inputs, columns)                   # numba JIT (startup)
        run_path(query_row(rows, 0), builder, context, fc, model, inputs, columns, reference=True)
        t_jit = time.time() - t0
    ready = process_memory()
    startup = {"imports": round(t_main - created, 2), "context": round(t_context, 2), "weights": round(t_weights, 2), "jit": round(t_jit, 2),
               "total": round(time.time() - created, 2)}
    passes = {}
    for p in PASSES:
        reference = p == "e2e_reference"
        clock = ModuleClock(model) if p == "breakdown" else None
        for j in range(n_warm):   # never timed
            run_path(query_row(rows, j), builder, context, fc, model, inputs, columns, reference)
        stages = {s: [] for s in (*STAGES, "total")}
        modules = {n: [] for n in clock.names} if clock else None
        tops, scores_all = [], []
        for j in timed_j:
            if clock:
                clock.reset()
            t, top, scores = run_path(query_row(rows, j), builder, context, fc, model, inputs, columns, reference)
            for s, v in t.items():
                stages[s].append(v)
            if clock:
                for n, v in clock.spent.items():
                    modules[n].append(v)
            tops.append(top)
            scores_all.append(scores)
        if clock:
            clock.remove()
        differs, largest = 0, 0.0   # outside the clock, after the pass
        for j, top, scores in zip(timed_j, tops, scores_all):
            want_top = sample["cached"][arm]["top5"][j]
            want = np.asarray(sample["cached"][arm]["scores"][j], dtype=np.float32)
            got = scores.numpy()
            differs += int(top.tolist() != want_top)
            largest = max(largest, float(np.max(np.abs(got - want))) if got.shape == want.shape and got.size else float("inf"))
        passes[p] = {"stages": {s: [round(v, 9) for v in vals] for s, vals in stages.items()},
                     "modules": None if modules is None else {n: [round(v, 9) for v in vals] for n, vals in modules.items()},
                     "top5_differs": differs, "largest_score_difference": largest,
                     "p50_ms": {s: round(1000 * float(np.median(vals)), 3) for s, vals in stages.items()}}
        log(f"   {arm} r{rnd} {p}: p50 total {passes[p]['p50_ms']['total']:.2f} ms, top-5 differs on {differs}")
    end = process_memory()
    rec = {"file": CONFIG.relative_to(ROOT).as_posix(), "utc": V2.utc(), "arm": arm, "round": rnd, "fit": key,
           "position_in_round": ORDERS[rnd].index(arm) if rnd < len(ORDERS) else None, "smoke": where.smoke, "idle_at_start": bool(idle),
           "queries": len(timed_j), "warm_up": n_warm, "startup_seconds": startup, "ready_wset": ready["wset"], "ready_peak_wset": ready["peak_wset"],
           "peak_wset": end["peak_wset"], "cpu_frequency": {"start": freq_start, "end": cpu_frequency()}, "gc_enabled": gc.isenabled(),
           "passes": passes, "sample_sha256": sha256_file(where.latency / "sample.json"), "placement": place,
           "git_head": DK.git_head(), "module_sha256": DK.module_shas()}
    atomic_json(where.latency / f"{arm}__r{rnd}.json", rec)
    return rec


def stage_memory_run(decl: dict, arm: str, rnd: int, idle: bool, where: Where = MAIN, log=log_utc) -> dict:
    """latency.memory (b): a fresh process with only the arm's seed-0 weights and the pre-packed batches (memory-mapped and
    touched before the ready point); one warm-up forward, the working set recorded, the forward over every batch, and the
    peak working set's increase over that point, beside the parameter and buffer bytes."""
    place = laptop_placement()
    verify_pins(decl)
    sample = read_json(where.latency / "sample.json")
    packed = where.latency / "packed_top128.pt"
    if sample is None or not packed.exists() or sha256_file(packed) != sample["packed_sha256"]:
        hard_stop("latency/packed_top128.pt is not the file latency-prepare pinned")
    key = DK.fit_key(arm, 0)
    with rebound(CheckedCompile(), where.eval):
        cfg, cfg_m3b, _ = V2.load_configs()
        inputs = V2.model_inputs(cfg, cfg_m3b)
        blob = torch.load(packed, map_location="cpu", weights_only=True, mmap=True)
        bank = RelationBank(blob["bank"]["embeddings"], dict(blob["bank"]["offsets"]), blob["bank"]["sha256"])
        model = load_fits(decl, inputs, bank, keys={key})[key]
    batches = [PackedBatch(**b) for b in blob["batches"]]
    touched = sum(int(touch(t)) for b in batches for t in batch_fields(b).values())   # every mapped page read before the ready point
    with torch.no_grad():
        model(V2.arm_view(model, batches[0], inputs))
        gc.collect()
        ready = process_memory()
        sampled = []
        for b in batches:
            model(V2.arm_view(model, b, inputs))
            sampled.append(process_memory()["wset"])
    end = process_memory()
    rec = {"file": CONFIG.relative_to(ROOT).as_posix(), "utc": V2.utc(), "arm": arm, "round": rnd, "fit": key, "smoke": where.smoke,
           "idle_at_start": bool(idle), "batches": len(batches), "ready_wset": ready["wset"], "ready_peak_wset": ready["peak_wset"],
           "peak_wset": end["peak_wset"], "increase_bytes": end["peak_wset"] - ready["wset"], "peak_moved": bool(end["peak_wset"] > ready["peak_wset"]),
           "sampled_increase_bytes": (max(sampled) - ready["wset"]) if sampled else 0, "touched_checksum": touched,
           "parameter_bytes": int(sum(p.numel() * p.element_size() for p in model.parameters())),
           "buffer_bytes": int(sum(b.numel() * b.element_size() for b in model.buffers())), "placement": place,
           "git_head": DK.git_head(), "module_sha256": DK.module_shas()}
    atomic_json(where.memory / f"{arm}__r{rnd}.json", rec)
    log(f"   memory {arm} r{rnd}: +{rec['increase_bytes'] / 2**20:.1f} MiB over the ready working set (peak moved: {rec['peak_moved']})")
    return rec


def power_scheme() -> str | None:
    """The active power scheme, read (powercfg /getactivescheme) and never changed."""
    try:
        return subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def laptop_state(log=log_utc, ac_wait_minutes: float = AC_WAIT_MINUTES, idle: dict = IDLE) -> dict:
    """latency.laptop_state before a timed or memory process: AC power (on battery the driver waits, then hard stops) and
    the system CPU below the threshold over a sample (the driver waits, polling, then runs and flags LAPTOP_NOT_IDLE)."""
    import psutil

    t0 = time.time()
    while True:
        bat = psutil.sensors_battery()
        plugged = True if bat is None else bool(bat.power_plugged)
        if plugged:
            break
        if time.time() - t0 > ac_wait_minutes * 60:
            hard_stop("the laptop has been on battery for 30 minutes before a timed process", battery_percent=getattr(bat, "percent", None))
        log("   on battery: waiting for AC power")
        time.sleep(30)
    t1, samples, ok = time.time(), [], False
    while True:
        cpu = psutil.cpu_percent(interval=idle["sample_seconds"])
        samples.append(round(float(cpu), 1))
        if cpu < idle["cpu_percent"]:
            ok = True
            break
        if time.time() - t1 > idle["wait_minutes"] * 60:
            break
        time.sleep(idle["poll_seconds"])
    bat = psutil.sensors_battery()
    return {"utc": V2.utc(), "ac_power": plugged, "battery_percent": None if bat is None else float(bat.percent),
            "cpu_percent_samples": samples, "idle": ok, "waited_seconds": round(time.time() - t0, 1)}


def stage_latency(decl: dict, where: Where = MAIN, log=log_utc) -> dict:
    """The driver: before every process the laptop state; the timed processes in the rotated order, then the memory
    processes; a stage over the ceiling is stopped and filed as a deviation (compute.ceiling)."""
    size = SIZES[where.smoke]
    if read_json(where.latency / "sample.json") is None:
        hard_stop("no latency/sample.json; latency-prepare runs first")
    idle = dict(IDLE) if not where.smoke else {**IDLE, "wait_minutes": 0.0}
    t_stage = time.time()
    driver = {"utc_start": V2.utc(), "power_scheme": power_scheme(), "processes": [], "deviations": []}
    script = str(Path(__file__).resolve())

    def launch(kind: str, arm: str, rnd: int) -> bool:
        state = laptop_state(log, idle=idle)
        remaining = CEILING_HOURS * 3600 - (time.time() - t_stage)
        cmd = [sys.executable, script, "--stage", kind, "--arm", arm, "--round", str(rnd), "--idle", str(int(state["idle"]))]
        if where.smoke:
            cmd.append("--smoke")
        log(f"== {kind} {arm} r{rnd} (idle {state['idle']}, cpu {state['cpu_percent_samples'][-1]}%)")
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, cwd=ROOT, timeout=max(60.0, remaining)).returncode
        except subprocess.TimeoutExpired:
            driver["deviations"].append(f"compute ceiling: the stage passed {CEILING_HOURS} h during {kind} {arm} r{rnd}; stopped")
            rc = None
        driver["processes"].append({"kind": kind, "arm": arm, "round": rnd, "laptop_state": state, "returncode": rc,
                                    "seconds": round(time.time() - t0, 1)})
        atomic_json(where.latency / "driver.json", driver)
        if rc not in (0,):
            if rc is not None:
                hard_stop(f"{kind} {arm} r{rnd} ended with return code {rc}")
            return False
        return True

    ok = True
    for rnd, order in enumerate(ORDERS[: size["rounds"]]):
        for arm in order:
            ok = ok and launch("latency-run", arm, rnd)
    for rnd, order in enumerate(ORDERS[: size["rounds"]]):
        for arm in order:
            ok = ok and launch("memory-run", arm, rnd)
    driver["utc_end"] = V2.utc()
    driver["seconds"] = round(time.time() - t_stage, 1)
    driver["complete"] = bool(ok)
    atomic_json(where.latency / "driver.json", driver)
    return driver


# ── stage: read ──────────────────────────────────────────────────────────────


def read_half(arrays: dict, mask: np.ndarray, metric: str) -> dict:
    """Per arm the seed mean per query; the contrasts and the share over one resample matrix (deploy_ck_2wiki's procedure)."""
    n = int(mask.sum())
    R = DK.resample_matrix(n)
    means = {a: DK.seed_mean(arrays, a, metric, mask) for a in ARMS}
    boot = {a: v[R].mean(axis=1) if n else np.zeros(DK.BOOT["resamples"]) for a, v in means.items()}
    out = {"queries": n, "arms": {a: round(float(v.mean()), 4) for a, v in means.items()},
           "per_seed": {a: [round(float(arrays[f"{DK.fit_key(a, s)}/{metric}"][mask].mean()), 4) for s in SEEDS] for a in ARMS},
           "contrasts": {}, "shares": {}}
    for c, (a, b) in CONTRASTS.items():
        out["contrasts"][c] = {"point": round(float(means[a].mean() - means[b].mean()), 4), "ci": DK.interval(boot[a] - boot[b]), "of": [a, b]}
    for s, ((a, b), (c, d)) in SHARES.items():
        den_b = boot[c] - boot[d]
        den = float(means[c].mean() - means[d].mean())
        readable = DK.interval(den_b)[0] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (boot[a] - boot[b]) / den_b
        out["shares"][s] = {"point": round(float(means[a].mean() - means[b].mean()) / den, 4) if den else None,
                            "ci": DK.interval(ratio[np.isfinite(ratio)]) if readable and np.isfinite(ratio).any() else None,
                            "readable": bool(readable), "of": [[a, b], [c, d]]}
    return out


def band_of(h: dict) -> str:
    """readings.retention.bands, exactly one."""
    if h["contrasts"]["gnn_gain"]["ci"][0] <= 0:
        return "GNN_GAIN_ABSENT"
    s = h["shares"]["full_share"]
    if s["readable"] and s["point"] is not None and s["ci"] is not None and s["point"] >= KEEP["share"] and s["ci"][0] >= KEEP["share_low"]:
        return "FULL_KEEPS"
    if h["contrasts"]["full_gain"]["ci"][0] > 0:
        return "FULL_PARTIAL"
    return "FULL_NO_GAIN"


def speed_band(ci: list[float]) -> str:
    """readings.speed.bands on a ratio's interval, exactly one."""
    if ci[1] <= SPEED["faster"]:
        return "E2E_FASTER"
    if ci[1] < SPEED["one"]:
        return "E2E_MARGINAL"
    return "E2E_NOT_FASTER"


def cluster_draws(q: int) -> np.ndarray:
    """statistics.latency: default_rng(0), 1000 resamples of query positions with replacement."""
    rng = np.random.default_rng(DK.BOOT["seed"])
    return np.stack([rng.integers(q, size=q) for _ in range(DK.BOOT["resamples"])])


def pooled_percentiles(M: np.ndarray, draws: np.ndarray | None = None) -> np.ndarray:
    """p50/p95/p99 of the pooled values of a (queries, rounds) matrix: on all values, or per resample over the drawn
    queries, each bringing its rounds -> (resamples, 3)."""
    if draws is None:
        return np.percentile(M.ravel(), QS)
    return np.percentile(M[draws].reshape(draws.shape[0], -1), QS, axis=1).T


def ratio_stats(num: np.ndarray, den: np.ndarray, draws: np.ndarray) -> dict:
    point = pooled_percentiles(num) / pooled_percentiles(den)
    boot = pooled_percentiles(num, draws) / pooled_percentiles(den, draws)
    return {f"p{q}": {"point": round(float(point[j]), 4), "ci": DK.interval(boot[:, j])} for j, q in enumerate(QS)}


def percentile_stats(M: np.ndarray, draws: np.ndarray) -> dict:
    point = 1000 * pooled_percentiles(M)
    boot = 1000 * pooled_percentiles(M, draws)
    return {f"p{q}_ms": {"point": round(float(point[j]), 3), "ci": [round(float(x), 3) for x in np.percentile(boot[:, j], [2.5, 97.5])]}
            for j, q in enumerate(QS)}


def latency_matrix(runs: dict, arm: str, rounds: int, pass_: str, what: str) -> np.ndarray:
    """(queries, rounds) seconds of one quantity: a path stage or the total, the graph part (the forward minus the
    node-local modules, breakdown pass) or the kernel module (breakdown pass, ck_full)."""
    cols = []
    for r in range(rounds):
        rec = runs[(arm, r)]["passes"][pass_]
        if what == "graph_part":
            local = sum(np.asarray(rec["modules"][n]) for n in NODE_LOCAL if n in rec["modules"])
            cols.append(np.asarray(rec["stages"]["forward"]) - local)
        elif what in NODE_LOCAL or what == "kernel":
            cols.append(np.asarray(rec["modules"][what]))
        else:
            cols.append(np.asarray(rec["stages"][what]))
    return np.stack(cols, axis=1)


def read_latency(runs: dict, rounds: int) -> dict:
    q = len(runs[(ARMS[0], 0)]["passes"]["e2e_fast"]["stages"]["total"])
    draws = cluster_draws(q)
    out = {"queries": q, "rounds": rounds, "e2e_fast": {}, "e2e_reference": {}, "forward": {}, "graph_part": {}, "kernel": None,
           "stages_p50_ms": {}, "ratios": {}}
    mats = {a: latency_matrix(runs, a, rounds, "e2e_fast", "total") for a in ARMS}
    fwd = {a: latency_matrix(runs, a, rounds, "e2e_fast", "forward") for a in ARMS}
    graph = {a: latency_matrix(runs, a, rounds, "breakdown", "graph_part") for a in ARMS}
    for a in ARMS:
        out["e2e_fast"][a] = percentile_stats(mats[a], draws)
        out["e2e_reference"][a] = percentile_stats(latency_matrix(runs, a, rounds, "e2e_reference", "total"), draws)
        out["forward"][a] = percentile_stats(fwd[a], draws)
        out["graph_part"][a] = percentile_stats(graph[a], draws)
        out["stages_p50_ms"][a] = {s: round(1000 * float(np.median(latency_matrix(runs, a, rounds, "e2e_fast", s))), 3) for s in STAGES}
    if "kernel" in (runs[("ck_full", 0)]["passes"]["breakdown"]["modules"] or {}):
        out["kernel"] = percentile_stats(latency_matrix(runs, "ck_full", rounds, "breakdown", "kernel"), draws)
    out["ratios"]["e2e_fast"] = {r: ratio_stats(mats[a], mats[b], draws) for r, (a, b) in RATIOS.items()}
    out["ratios"]["forward"] = {r: ratio_stats(fwd[a], fwd[b], draws) for r, (a, b) in RATIOS.items()}
    out["ratios"]["graph_part"] = {"ck_full/gnn": ratio_stats(graph["ck_full"], graph["gnn"], draws)}
    return out


def read_memory(mems: dict, rounds: int) -> dict:
    out = {}
    for a in ARMS:
        recs = [mems[(a, r)] for r in range(rounds)]
        out[a] = {"median_increase_bytes": float(np.median([m["increase_bytes"] for m in recs])),
                  "median_sampled_increase_bytes": float(np.median([m["sampled_increase_bytes"] for m in recs])),
                  "peak_moved_rounds": int(sum(m["peak_moved"] for m in recs)),
                  "parameter_bytes": recs[0]["parameter_bytes"], "buffer_bytes": recs[0]["buffer_bytes"]}
    g = out["gnn"]["median_increase_bytes"]
    out["ratio_ck_full/gnn"] = round(out["ck_full"]["median_increase_bytes"] / g, 4) if g > 0 else None
    return out


def flags_of(halves: dict, bands: dict, latency: dict, memory: dict, runs: dict, mems: dict) -> tuple[str, list[str], dict]:
    """readings.speed and readings.flags from the numbers, in the declared order."""
    p = halves[HALVES[0]][READ_METRICS[0]]
    reading = bands[HALVES[0]][READ_METRICS[0]]
    e2e = latency["ratios"]["e2e_fast"]["ck_full/gnn"]
    speed = speed_band(e2e["p50"]["ci"])
    tails = {q: speed_band(e2e[q]["ci"]) for q in ("p95", "p99")}
    forward = speed_band(latency["ratios"]["forward"]["ck_full/gnn"]["p50"]["ci"])
    raised = set()
    if p["contrasts"]["full_vs_gnn"]["ci"][1] >= 0:
        raised.add("FULL_NOT_BELOW_GNN")
    if bands["V2_GATE"][READ_METRICS[0]] != reading:
        raised.add("GATE_HALF_DIFFERS")
    if bands[HALVES[0]][READ_METRICS[1]] != reading:
        raised.add("FC5_DIFFERS")
    if e2e["p50"]["ci"][0] > SPEED["one"]:
        raised.add("E2E_SLOWER")
    if any(b != speed for b in tails.values()):
        raised.add("TAIL_DIFFERS")
    if forward != speed:
        raised.add("FORWARD_ONLY_DIFFERS")
    if memory["ck_full"]["median_increase_bytes"] > memory["gnn"]["median_increase_bytes"]:
        raised.add("MEMORY_ABOVE_GNN")
    if any(pa["top5_differs"] for r in runs.values() for pa in r["passes"].values()):
        raised.add("E2E_PATH_DIFFERS")
    if not all(r["idle_at_start"] for r in (*runs.values(), *mems.values())):
        raised.add("LAPTOP_NOT_IDLE")
    return speed, [f for f in FLAGS if f in raised], {"tails": tails, "forward_only": forward}


def fmt_share(s: dict) -> str:
    if not s["readable"] or s["ci"] is None:
        return f"{s['point']:.3f} (not read: the denominator's interval reaches 0)" if s["point"] is not None else "not read"
    return f"{s['point']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"


def pareto_statement(h: dict, latency: dict, memory: dict) -> str:
    """readings.pareto_statement, filled from the numbers."""
    s = h["shares"]["full_share"]
    share = f"{s['point']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]" if s["readable"] and s["ci"] is not None else "an unread share (the GNN's gain is not read)"
    r = latency["ratios"]["e2e_fast"]["ck_full/gnn"]
    m = memory["ratio_ck_full/gnn"]
    twin = latency["ratios"]["e2e_fast"]["twin/gnn"]
    return (f"ck_full keeps {share} of the GNN's recall@5 gain over the twin on V2_HELD_CONFIRMATION at {r['p50']['point']:.3f} "
            f"[{r['p50']['ci'][0]:.3f}, {r['p50']['ci'][1]:.3f}] of the GNN's end-to-end p50 latency ({r['p95']['point']:.3f} at p95, "
            f"{r['p99']['point']:.3f} at p99) and {'n/a' if m is None else f'{m:.3f}'} of its model-side memory increase; laptop CPU, 8 threads, "
            f"a batch of one, the fast compiler wired in. The twin, the no-propagation floor: {twin['p50']['point']:.3f} of the GNN's p50 "
            f"({twin['p95']['point']:.3f} at p95, {twin['p99']['point']:.3f} at p99).")


def stage_read(decl: dict, where: Where = MAIN, log=log_utc) -> dict:
    rec = read_json(where.eval / f"{NAME}.json")
    if rec is None:
        hard_stop("read: no eval record")
    size = SIZES[where.smoke]
    rounds = size["rounds"]
    runs = {(a, r): read_json(where.latency / f"{a}__r{r}.json") for a in ARMS for r in range(rounds)}
    mems = {(a, r): read_json(where.memory / f"{a}__r{r}.json") for a in ARMS for r in range(rounds)}
    missing = [f"{k[0]}__r{k[1]}" for d in (runs, mems) for k, v in d.items() if v is None]
    if missing:
        hard_stop("read: timed or memory records are missing", missing=missing)
    with np.load(where.eval / f"{NAME}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    half = arrays["half"]
    masks = {"V2_HELD_CONFIRMATION": ~half, "V2_GATE": half, "whole": np.ones(half.size, dtype=bool)}
    halves = {h: {m: read_half(arrays, mask, m) for m in READ_METRICS} for h, mask in masks.items()}
    bands = {h: {m: band_of(halves[h][m]) for m in READ_METRICS} for h in halves}
    reading = bands[HALVES[0]][READ_METRICS[0]]
    latency = read_latency(runs, rounds)
    memory = read_memory(mems, rounds)
    speed, flags, other_bands = flags_of(halves, bands, latency, memory, runs, mems)
    comp = DK.compiled_check(arrays, list(rec["compiled_form"]))   # an exactness check of ck_full's moments form, never timed
    for k, v in comp.items():
        v["max_abs_score_difference"] = rec["compiled_form"][k]["max_abs_score_difference"]
    out = {"utc": V2.utc(), "dataset": NAME, "smoke": where.smoke, "reading": reading, "speed": speed, "flags": flags,
           "other_speed_bands": other_bands, "pareto_statement": pareto_statement(halves[HALVES[0]][READ_METRICS[0]], latency, memory),
           "primary": {"half": HALVES[0], "metric": READ_METRICS[0], "co_read": READ_METRICS[1]}, "bands": bands, "halves": halves,
           "latency": latency, "memory": memory, "compiled_form": comp,
           "path_checks": {f"{a}__r{r}": {p: runs[(a, r)]["passes"][p]["top5_differs"] for p in PASSES} for a, r in runs},
           "eval_sha256": sha256_file(where.eval / f"{NAME}.json"), "arrays_sha256": sha256_file(where.eval / f"{NAME}.npz"),
           "git_head": DK.git_head(), "module_sha256": DK.module_shas()}
    atomic_json(where.read, out)
    log(f"read: {reading} and {speed}; flags {flags}")
    log(f"   {out['pareto_statement']}")
    return out


# ── stage: doc, file ─────────────────────────────────────────────────────────


def stage_doc(decl: dict, where: Where = MAIN, log=log_utc) -> Path:
    """record.json from read.json, the eval record and the latency records (no arithmetic), then the document."""
    read, ev, sample = read_json(where.read), read_json(where.eval / f"{NAME}.json"), read_json(where.latency / "sample.json")
    driver = read_json(where.latency / "driver.json")
    if read is None or ev is None or sample is None or driver is None:
        raise SystemExit("doc: read.json, the eval record, latency/sample.json and latency/driver.json come first")
    rounds = SIZES[where.smoke]["rounds"]
    procs = {}
    for a in ARMS:
        for r in range(rounds):
            run, mem = read_json(where.latency / f"{a}__r{r}.json"), read_json(where.memory / f"{a}__r{r}.json")
            procs[f"{a}__r{r}"] = {"startup_seconds": run["startup_seconds"], "ready_wset": run["ready_wset"], "peak_wset": run["peak_wset"],
                                   "cpu_frequency": run["cpu_frequency"], "idle_at_start": run["idle_at_start"],
                                   "memory": {k: mem[k] for k in ("increase_bytes", "peak_moved", "sampled_increase_bytes", "ready_wset", "idle_at_start")}}
    record = {"file": CONFIG.relative_to(ROOT).as_posix(), "utc": V2.utc(), "smoke": where.smoke, "reading": read["reading"], "speed": read["speed"],
              "flags": read["flags"], "other_speed_bands": read["other_speed_bands"], "pareto_statement": read["pareto_statement"],
              "primary": read["primary"], "bands": read["bands"], "halves": read["halves"], "latency": read["latency"], "memory": read["memory"],
              "compiled_form": read["compiled_form"], "path_checks": read["path_checks"], "processes": procs,
              "driver": {k: driver.get(k) for k in ("power_scheme", "seconds", "complete", "deviations")},
              "eval": {"queries": ev["queries"], "ids_sha256": ev["ids_sha256"], "halves": ev["halves"], "ceiling_recall@5": ev["ceiling_as_compiled"].get("recall_ceiling@5"),
                       "m3b_fixed_rrf_agreement": ev["m3b_fixed_rrf_agreement"]["ok"], "mrr_audit_ok": all(v["ok"] for v in ev["mrr_audit"].values()),
                       "seconds": ev["seconds"], "fast_compile": ev["fast_compile"], "batch_check": ev.get("batch_check"),
                       "host_agreement": ev["host_agreement"], "placement": ev["placement"], "rebinding": ev["rebinding"]},
              "sample": {"timed": len(sample["timed"]), "warm": len(sample["warm"]), "pools_rebuilt_equal": sample["pools_rebuilt_equal"],
                         "fast_compile": sample["fast_compile"], "first_stage_sha256": sample["first_stage_sha256"], "packed_sha256": sample["packed_sha256"]},
              "read_sha256": sha256_file(where.read), "eval_sha256": sha256_file(where.eval / f"{NAME}.json")}
    atomic_json(where.record, record)
    where.doc.parent.mkdir(parents=True, exist_ok=True)
    where.doc.write_text(LF.join(doc_lines(decl, record)) + LF, encoding="utf-8", newline=LF)
    log(f"doc: {DK.shown(where.record)} and {DK.shown(where.doc)}")
    return where.doc


def _ci(c: dict, fmt: str = "+.4f") -> str:
    return f"{c['point']:{fmt}} [{c['ci'][0]:{fmt}}, {c['ci'][1]:{fmt}}]"


def doc_lines(decl: dict, rec: dict) -> list[str]:
    ph, pm = rec["primary"]["half"], rec["primary"]["metric"]
    L = ["# ck_full end to end against the GNN on 2wiki (configs/deploy_ck_full_2wiki.yaml)", ""]
    if rec["smoke"]:
        L += ["**SMOKE: a shard and 16 sampled queries, never read.**", ""]
    L += [f"**Retention: `{rec['reading']}`** on {ph} {pm}; **speed: `{rec['speed']}`** on the end-to-end p50 ratio ck_full / gnn; flags: "
          + (", ".join(f"`{f}`" for f in rec["flags"]) or "none") + ".", "",
          rec["pareto_statement"], "",
          "ck_full is a compressed, query-conditioned one-hop message-passing approximation (its kernel is learned propagation). Every arm "
          "is deploy_ck_2wiki's sealed fit, unchanged; the retention was scored again on this laptop's CPU at 8 threads with the fast feature "
          "compiler wired in and compared with the reference compile on every query. Every number here is a laptop number on 2wiki's dev "
          "population; none is set beside a host number.", "",
          "## Retention (seed means per query, then the mean)", ""]
    for h in HALVES:
        L += [f"### {h} ({rec['halves'][h][pm]['queries']} queries)", "", "| arm | parameters | " + " | ".join(READ_METRICS) + " | per-seed R@5 |",
              "|---|---:|" + "---:|" * len(READ_METRICS) + "---|"]
        for a in ARMS:
            L.append(f"| {a} | {decl['arms'][a]['parameters']:,} | " + " | ".join(f"{rec['halves'][h][m]['arms'][a]:.4f}" for m in READ_METRICS)
                     + " | " + ", ".join(f"{v:.4f}" for v in rec["halves"][h]["recall@5"]["per_seed"][a]) + " |")
        L += ["", "| contrast | " + " | ".join(READ_METRICS) + " |", "|---|" + "---|" * len(READ_METRICS)]
        for c, (a, b) in CONTRASTS.items():
            L.append(f"| {c} ({a} - {b}) | " + " | ".join(_ci(rec["halves"][h][m]["contrasts"][c]) for m in READ_METRICS) + " |")
        L.append("| full_share | " + " | ".join(fmt_share(rec["halves"][h][m]["shares"]["full_share"]) for m in READ_METRICS) + " |")
        L += ["", "Bands: " + ", ".join(f"{m} `{rec['bands'][h][m]}`" for m in READ_METRICS) + ".", ""]
    lat = rec["latency"]
    L += ["## Latency (laptop CPU, 8 threads, a batch of one)", "",
          f"{lat['queries']} held queries x {lat['rounds']} rounds per arm, each arm in a fresh process per round in a rotated order; "
          "percentiles of the pooled per-query totals, 95% cluster-bootstrap intervals (a drawn query brings its rounds).", "",
          "| arm | e2e p50 ms | e2e p95 ms | e2e p99 ms | forward p50 ms | graph part p50 ms | reference-compile e2e p50 ms |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        e = lat["e2e_fast"][a]
        L.append(f"| {a} | {_ci(e['p50_ms'], '.2f')} | {_ci(e['p95_ms'], '.2f')} | {_ci(e['p99_ms'], '.2f')} | {lat['forward'][a]['p50_ms']['point']:.2f} "
                 f"| {lat['graph_part'][a]['p50_ms']['point']:.2f} | {lat['e2e_reference'][a]['p50_ms']['point']:.2f} |")
    if lat.get("kernel"):
        k = lat["kernel"]
        L += ["", f"ck_full's kernel module alone (the query-dependent kernel built and applied): p50 {_ci(k['p50_ms'], '.3f')} ms, "
                  f"p95 {k['p95_ms']['point']:.3f}, p99 {k['p99_ms']['point']:.3f}."]
    L += ["", "| ratio | p50 | p95 | p99 |", "|---|---|---|---|"]
    for r in RATIOS:
        x = lat["ratios"]["e2e_fast"][r]
        L.append(f"| e2e {r} | {_ci(x['p50'], '.3f')} | {_ci(x['p95'], '.3f')} | {_ci(x['p99'], '.3f')} |")
    for r in RATIOS:
        x = lat["ratios"]["forward"][r]
        L.append(f"| forward {r} | {_ci(x['p50'], '.3f')} | {_ci(x['p95'], '.3f')} | {_ci(x['p99'], '.3f')} |")
    x = lat["ratios"]["graph_part"]["ck_full/gnn"]
    L.append(f"| graph part ck_full/gnn | {_ci(x['p50'], '.3f')} | {_ci(x['p95'], '.3f')} | {_ci(x['p99'], '.3f')} |")
    L += ["", "Median per path stage (e2e_fast, ms):", "", "| arm | " + " | ".join(STAGES) + " |", "|---|" + "---:|" * len(STAGES)]
    for a in ARMS:
        L.append(f"| {a} | " + " | ".join(f"{lat['stages_p50_ms'][a][s]:.3f}" for s in STAGES) + " |")
    L += ["", f"Other speed bands: p95/p99 {rec['other_speed_bands']['tails']}, forward alone `{rec['other_speed_bands']['forward_only']}`.", "",
          "## Memory", "", "| arm | model-side increase MiB (median over rounds) | sampled increase MiB | peak moved (rounds) | parameter KiB |",
          "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        m = rec["memory"][a]
        L.append(f"| {a} | {m['median_increase_bytes'] / 2**20:.2f} | {m['median_sampled_increase_bytes'] / 2**20:.2f} | {m['peak_moved_rounds']} "
                 f"| {m['parameter_bytes'] / 1024:.1f} |")
    L += ["", "| process | startup s | ready working set MiB | peak working set MiB | idle at start |", "|---|---:|---:|---:|---|"]
    for k, p in rec["processes"].items():
        L.append(f"| {k} | {p['startup_seconds']['total']:.1f} | {p['ready_wset'] / 2**20:.0f} | {p['peak_wset'] / 2**20:.0f} | {p['idle_at_start']} |")
    e = rec["eval"]
    L += ["", "## Checks", "",
          f"- Fast compile: {e['fast_compile']['queries_checked']} eval queries and {rec['sample']['fast_compile']['queries_checked']} sampled "
          "queries compiled by both compilers, bit-identical on every array of every query.",
          f"- Batches (amendment 2): {e['batch_check']}." if e.get("batch_check") else "- Batches (amendment 2): not compared (smoke shard).",
          f"- Pools rebuilt from the first-stage lists equal the prepared pools on all {rec['sample']['pools_rebuilt_equal']} sampled and warm-up queries.",
          f"- Timed paths' top-5 against the cached pass: {rec['path_checks']}.",
          f"- Eval population {e['queries']} queries (ids {e['ids_sha256'][:12]}); recall ceiling@5 {e['ceiling_recall@5']:.4f}; "
          f"M3B fixed-rrf agreement {e['m3b_fixed_rrf_agreement']}; MRR audit {e['mrr_audit_ok']}.",
          "- Host agreement (systems count, not read): " + ", ".join(f"{k} {v['queries_where_a_metric_differs']}" for k, v in e["host_agreement"].items()) + "."]
    for k, v in rec["compiled_form"].items():
        L.append(f"- Compiled form of `{k}` (an exactness check, never timed): largest score difference {v['max_abs_score_difference']:.2e}, "
                 f"queries where a metric differs {v['queries_where_a_metric_differs']}.")
    L += [f"- Power scheme (read, never changed): {rec['driver']['power_scheme']}; driver deviations: {rec['driver']['deviations'] or 'none'}.",
          "", "## Wording", "", str(decl["readings"]["wording"]).strip(), ""]
    return L


def stage_file(decl: dict, date: str, commit: str, extra: dict | None = None, log=log_utc) -> None:
    """run_record_deploy_ck_full_2wiki_<date> appended after the code-identity check; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    key = f"run_record_deploy_ck_full_2wiki_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    if DK.committed_lf_sha(commit, "scripts/deploy_ck_full_2wiki.py") is None:
        raise SystemExit(f"{commit} does not hold scripts/deploy_ck_full_2wiki.py")
    rec = read_json(MAIN.record)
    if rec is None or rec.get("smoke"):
        raise SystemExit("file: record.json is missing or is the smoke's")
    jobs = {"score": read_json(MAIN.eval / f"{NAME}.json")["module_sha256"], "latency-prepare": read_json(MAIN.latency / "sample.json")["module_sha256"],
            "read": read_json(MAIN.read)["module_sha256"]}
    for a in ARMS:
        for r in range(SIZES[False]["rounds"]):
            jobs[f"latency/{a}__r{r}"] = read_json(MAIN.latency / f"{a}__r{r}.json")["module_sha256"]
            jobs[f"memory/{a}__r{r}"] = read_json(MAIN.memory / f"{a}__r{r}.json")["module_sha256"]
    bad = DK.code_problems(jobs, lambda path: DK.committed_lf_sha(commit, path))
    if bad:
        hard_stop("identical_code: the processes' recorded module sha256 values are not one set equal to the committed files", problems=bad)
    p = rec["halves"][rec["primary"]["half"]][rec["primary"]["metric"]]
    lat = rec["latency"]
    run = {"utc": V2.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "code_commit": commit,
           "reading": rec["reading"], "speed": rec["speed"], "flags": rec["flags"], "pareto_statement": rec["pareto_statement"],
           "arms_primary": p["arms"], "contrasts_primary": {c: {"point": v["point"], "ci": v["ci"]} for c, v in p["contrasts"].items()},
           "full_share_primary": p["shares"]["full_share"], "bands": rec["bands"],
           "e2e_ratios": {r: {q: lat["ratios"]["e2e_fast"][r][q] for q in ("p50", "p95", "p99")} for r in RATIOS},
           "e2e_p50_ms": {a: lat["e2e_fast"][a]["p50_ms"]["point"] for a in ARMS},
           "memory_ratio_ck_full/gnn": rec["memory"]["ratio_ck_full/gnn"],
           "identical_code": f"{len({q for s in jobs.values() for q in s})} module paths over {len(jobs)} processes, one sha256 each, equal to the committed files",
           "placement": "every stage on the laptop CPU at 8 threads (BLAS, OpenMP, numba, torch)", "held_half_read": True,
           "record_sha256": sha256_file(MAIN.record), "document": DOC.relative_to(ROOT).as_posix(), "document_sha256": lf_sha256(DOC)}
    if extra:
        run.update(extra)
    block = yaml.safe_dump(clean({key: run}), sort_keys=False, width=160, allow_unicode=True)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8", newline=LF)   # the declaration stays LF
    log(f"filed {key}")


def stage_smoke(decl: dict, log=log_utc) -> None:
    """tests: the smoke on the real 2wiki data, read-only: a shard of the score pass, latency-prepare on 16 sampled queries,
    one round of timed and memory processes, then read and doc on those records (under outputs/deploy_ck_full_2wiki/smoke)."""
    stage_score(decl, SMOKE, log)
    stage_latency_prepare(decl, SMOKE, log)
    stage_latency(decl, SMOKE, log)
    stage_read(decl, SMOKE, log)
    stage_doc(decl, SMOKE, log)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ck_full end to end against the GNN on 2wiki (configs/deploy_ck_full_2wiki.yaml)")
    ap.add_argument("--stage", required=True, choices=["smoke", "score", "latency-prepare", "latency", "latency-run", "memory-run", "read", "doc", "file"])
    ap.add_argument("--arm", choices=list(ARMS), default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--idle", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="latency-run / memory-run: the smoke's files")
    ap.add_argument("--date", default=None)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--extra", default=None, help="a JSON file of run-record fields (deviations)")
    args = ap.parse_args(argv)
    torch.set_num_threads(THREADS)
    decl = load_declaration()
    where = SMOKE if args.smoke else MAIN
    if args.stage == "smoke":
        stage_smoke(decl)
    elif args.stage == "score":
        stage_score(decl)
    elif args.stage == "latency-prepare":
        stage_latency_prepare(decl)
    elif args.stage == "latency":
        stage_latency(decl)
    elif args.stage in ("latency-run", "memory-run"):
        if args.arm is None or args.round is None:
            raise SystemExit(f"--stage {args.stage} needs --arm and --round")
        fn = stage_latency_run if args.stage == "latency-run" else stage_memory_run
        fn(decl, args.arm, args.round, bool(args.idle), where)
    elif args.stage == "read":
        stage_read(decl)
    elif args.stage == "doc":
        stage_doc(decl)
    else:
        if not args.date or not args.commit:
            raise SystemExit("--stage file needs --date and --commit")
        extra = json.loads(Path(args.extra).read_text(encoding="utf-8")) if args.extra else None
        stage_file(decl, args.date, args.commit, extra)


if __name__ == "__main__":
    main()
