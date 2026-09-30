"""MP-Approx six-dataset base, work 2 (configs/mp_approx_six_base.yaml#work_2_scoring_pass): level 0's scoring pass with
the six-dataset pair, and the filing of both works.

    python scripts/mp_approx_six_base_score.py --stage fits-record --date 2026_10_01   # fits_record_<date>, once work 1 is done
    python scripts/mp_approx_six_base_score.py --stage equivalence --dataset metaqa     # the copy against level 0, trio pair
    python scripts/mp_approx_six_base_score.py --stage score --dataset hotpotqa         # one dataset per process, 4 threads
    python scripts/mp_approx_six_base_score.py --stage run                              # the three equivalences, then the six scores
    python scripts/mp_approx_six_base_score.py --stage status
    python scripts/mp_approx_six_base_score.py --stage file --date 2026_10_01 --commit <sha>

On the host (amendment_2_2026_09_30_host_placement): --host puts the verified mirror in place of the package, in memory
only; --shard i/n scores every n-th chunk, and one later run without --shard assembles; --stage verify re-reads every
array there, since the sidecars stay on the host.

    python scripts/mp_approx_six_base_score.py --host --stage equivalence --dataset squad
    python scripts/mp_approx_six_base_score.py --host --stage score --dataset musique --shard 0/4
    python scripts/mp_approx_six_base_score.py --host --stage score --dataset musique    # assembles the shards' chunks
    python scripts/mp_approx_six_base_score.py --host --stage verify

The pair: T_k = u_mlp_v2_mix__H128__six__s{k} (work 1), G_k = u_gnn_v2_ef__H128__six__s{k} (universal-v2 stage 2) and
G0_k = G_k with message_passing False for one forward. Measurement only: nothing here becomes a retriever, a feature, a
teacher or a selection criterion, and the GNN's outputs are kept only as the ladder's targets.

This is work 2's own script, apart from scripts/mp_approx_six_base.py, so the joint fits of work 1, which run while it is
written, never import it. Level 0's code is imported unchanged wherever it is parameterised by paths: the copied attention
and the taps, the no-edge forward, the analysis set and the per-query helpers, the id rules, assemble and the sidecar
reader. Level 0's stage_score is not (its pair, stored arrays, rows, pins and thread count are module constants), so it
is copied here as score_pair with those passed in as a Pair, and the equivalence stage holds the copy equal to level 0's
on level 0's trio pair, array for array.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: the BLAS pools are fixed before numpy loads
    # the equivalence run is level 0's scoring pass (level 0 ran it at 6); the scoring pass is stage 2's eval, whose
    # BLAS pools universal_v2_six.py sets (M3B_BLAS_THREADS, else 2)
    _BLAS = "6" if "equivalence" in sys.argv else os.environ.get("M3B_BLAS_THREADS", "2")
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _BLAS

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_six_base as SB  # noqa: E402  (work 1, imported unchanged)
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402

V2, U6 = SB.V2, SB.U6
CONFIG, OUT, FITS = SB.CONFIG, SB.OUT, SB.FITS
EQUIV = OUT / "equivalence"
STAGE2_EVAL = U6.SIX_EVAL
SCRIPT_REL = "scripts/mp_approx_six_base_score.py"
LEVEL_0_REL = "scripts/mp_approx_l0.py"
SEEDS = SB.SEEDS
DATASETS = SB.DATASETS
TRIO = L0.DATASETS
SCORE_THREADS = 4   # placement.threads: work 2 at 4, the thread count of the stage-2 eval records
MAX_EPOCHS = 6      # the frozen rule's (universal_v2_six.training_rule_six)
EQUIVALENCE_LIMITS = {"metaqa": 24, "2wiki": 240, "squad": 500}   # the first N of level 0's rows: two chunks on each
RUN_ORDER = ("squad", "2wiki", "hotpotqa", "webqsp", "metaqa", "musique")   # the cheap populations first
GAP_METRICS = ("recall@5", "hit@1", "full_coverage@5")
LF = chr(10)
HOST_BLOCK = "amendment_2_2026_09_30_host_placement"
MIRROR_CONFIG = ROOT / "configs" / "host_mirror_six.yaml"
HOST_VERIFY = "verify_host.json"
PLACEMENT = {"where": "laptop"}   # --host makes it the host (amendment 2); filed with every record this process writes
SIDECAR_FETCHED = ("query.npy", "q_row.npy", "q_fold.npy", "q_metrics.npy")   # what the file stage reads when the arrays stay on the host

L0.HARD_STOP_DIR[0] = OUT   # a level-0 helper's hard stop (the no-edge forward's) lands here, never under outputs/mp_approx_l0

INTEGRITY = {
    "six": ("rank_metrics of G_k on the full pool equal stage 2's stored per-query values of all 14 metrics on every scored "
            "query (outputs/universal_v2/six/eval/<name>.npz); T_k has no stored counterpart, its full-pool metrics are "
            "filed in q_metrics and nothing is checked against them; the copied attention equals the cell's message at every "
            "step of every chunk (atol 1e-5, rtol 1e-5); within U_q the nine top-20 metrics and gold_in_pool of all nine "
            "functions equal their full-pool values; message_passing and every state_dict were restored after each no-edge "
            "forward"),
    "trio": ("level 0's: rank_metrics of T_k and G_k on the full pool equal the stored per-query values of all 14 metrics on "
             "every scored query; the copied attention equals the cell's message at every step of every chunk (atol 1e-5, "
             "rtol 1e-5); within U_q the nine top-20 metrics and gold_in_pool of all nine functions equal their full-pool "
             "values; message_passing and every state_dict were restored after each no-edge forward"),
}


def log_utc(msg: str) -> None:
    print(f"[{L0.utc()}] {msg}", flush=True)


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to outputs/mp_approx_six_base/hard_stop.json and nothing further runs."""
    SB.hard_stop(message, **{k: L0._jsonable(v) for k, v in evidence.items()})


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def blas_env() -> dict:
    return {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}


# ── the host (amendment_2_2026_09_30_host_placement) ─────────────────────────


def host_mode(decl: dict, log=print) -> dict:
    """amendment_2.mirror.how_it_is_read: the root is the one configs/host_mirror_six.yaml names and the amendment
    declares, and each pinned verify record is VERIFIED there with the declared freeze and the loader imported from
    the mirror, the records covering the six datasets; anything else refuses the process before it opens anything.
    universal_v2_run.load_configs is then wrapped in this process, so level 0's stage_score and score_pair both get
    substrate.package_root = the mirror. No config file is edited; open_package's freeze check still runs."""
    block = decl.get(HOST_BLOCK)
    if not block:
        raise SystemExit(f"--host: {HOST_BLOCK} is not filed in {CONFIG.name}; work 2 runs on the laptop")
    m = block["mirror"]
    root = yaml.safe_load(MIRROR_CONFIG.read_text(encoding="utf-8"))["host"]["mirror_root"]
    if root != m["root"]:
        hard_stop("--host: the host mirror root differs from the declared one", declared=m["root"], config=root)
    served = (Path(root) / "data" / "final_canonical").as_posix()
    covered = []
    for rel, want in m["verify_records"].items():
        path = ROOT / rel
        if not path.exists() or SB.sha256_file(path) != want:
            hard_stop(f"--host: {rel} is not the pinned verify record", path=rel)
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not (rec.get("status") == "VERIFIED" and rec.get("freeze_matches_declared") is True
                and rec.get("loader_imported_from_mirror") is True and Path(rec.get("mirror", "")).as_posix() == served
                and rec.get("freeze_RECORD_SHA256") == m["freeze_RECORD_SHA256"]):
            hard_stop(f"--host: {rel} is not VERIFIED at the declared root", record=rec)
        covered += list(rec.get("datasets", []))
    if sorted(covered) != sorted(DATASETS):
        hard_stop("--host: the verify records do not cover the six datasets once each", covered=covered)
    original = V2.load_configs

    def load_configs_on_the_mirror(*args, **kwargs):
        cfg, cfg_m3b, cfg_h = original(*args, **kwargs)
        cfg_m3b["substrate"]["package_root"] = str(Path(root))   # in memory only
        return cfg, cfg_m3b, cfg_h

    V2.load_configs = load_configs_on_the_mirror
    PLACEMENT.clear()
    PLACEMENT.update({"where": "host", "node": platform.node(), "mirror_root": root, "amendment": HOST_BLOCK})
    log(f"host: the mirror at {root} in place of the package, in memory; {len(m['verify_records'])} verify records VERIFIED")
    return PLACEMENT


def parse_shard(text: str | None) -> tuple[int, int] | None:
    """--shard i/n (amendment_2.shards): 0 <= i < n, n >= 2."""
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
    """The chunks a process scores: all of them, or shard i's ci with ci mod n = i."""
    return list(range(n_chunks)) if shard is None else [ci for ci in range(n_chunks) if ci % shard[1] == shard[0]]


# ── the pair ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pair:
    """One scored pair: the fit-key templates of T_k and G_k, the functions whose full-pool metrics must equal stored
    values, and the scoring threads. The plumbing below reads the name: "trio" is level 0's own pair, read through level
    0's functions; "six" is this file's."""
    name: str
    twin: str
    gnn: str
    stored: tuple
    threads: int

    def keys(self) -> dict:
        return {f"{fam}{k}": tmpl.format(k=k) for fam, tmpl in (("twin", self.twin), ("gnn", self.gnn)) for k in SEEDS}


SIX = Pair("six", f"{SB.ARM}__H{V2.HIDDEN}__six__s{{k}}", f"{U6.ARM}__H{V2.HIDDEN}__six__s{{k}}",
           tuple(f"gnn{k}" for k in SEEDS), SCORE_THREADS)
TRIO_PAIR = Pair("trio", L0.TWIN, L0.GNN, tuple(L0.STORED_FUNCS), L0.SCORE_THREADS)


# ── work 2's pins and preconditions ──────────────────────────────────────────


def fits_record_block(decl: dict) -> dict | None:
    keys = sorted(k for k in decl if k.startswith("fits_record_"))
    return decl[keys[-1]] if keys else None


def preconditions(decl: dict, equiv_dir: Path | None = None, need_equivalence: bool = True) -> list[str]:
    """What work 2 needs filed before a scoring pass starts; a missing one refuses the stage (nothing is written)."""
    equiv_dir = EQUIV if equiv_dir is None else equiv_dir
    s2 = decl["inputs"]["stage_2_eval"]
    missing = []
    if sorted(s2.get("pins") or {}) != sorted(DATASETS):
        missing.append("inputs.stage_2_eval.pins for the six datasets")
    if LEVEL_0_REL not in (s2.get("level_0_code_lf") or {}):
        missing.append(f"inputs.stage_2_eval.level_0_code_lf[{LEVEL_0_REL}]")
    block = fits_record_block(decl)
    if block is None or sorted(block.get("fits") or {}) != sorted(SB.fit_key(k) for k in SEEDS):
        missing.append("fits_record_<date> with the three twin fits (execution: work 2 follows work 1's record)")
    if need_equivalence:
        for name in TRIO:
            rec = V2.read_json(equiv_dir / f"{name}.json")
            if rec is None or rec.get("equal") is not True:
                missing.append(f"equivalence/{name}.json with equal true")
    return missing


def pin_differences(decl: dict, fits_dir: Path | None = None) -> list[str]:
    """Filed pins whose file differs: stage 2's eval arrays and id lists, level 0's code (LF) and the twin's weights."""
    fits_dir = FITS if fits_dir is None else fits_dir
    s2, bad = decl["inputs"]["stage_2_eval"], []
    for name, files in (s2.get("pins") or {}).items():
        for kind, path in (("npz", STAGE2_EVAL / f"{name}.npz"), ("query_ids", STAGE2_EVAL / f"{name}_query_ids.json")):
            if not path.exists() or SB.sha256_file(path) != files[kind]:
                bad.append(path.relative_to(ROOT).as_posix())
    for rel, want in (s2.get("level_0_code_lf") or {}).items():
        if not (ROOT / rel).exists() or SB.lf_sha256(ROOT / rel) != want:
            bad.append(rel)
    block = fits_record_block(decl) or {}
    for key, row in (block.get("fits") or {}).items():
        path = fits_dir / f"{key}.pt"
        if not path.exists() or SB.sha256_file(path) != row.get("state_sha256"):
            bad.append(path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path))
    return bad


def verify_six(decl: dict) -> None:
    """At the start and the end of every scoring process: work 1's pins (frozen code, the stage-2 checkpoints, the
    caches, the contract), then work 2's."""
    missing = preconditions(decl)
    if missing:
        raise SystemExit(f"work 2 does not start: {missing} not filed")
    SB.verify_pins(decl)
    bad = pin_differences(decl)
    if bad:
        hard_stop(f"work 2's pinned inputs differ from configs/mp_approx_six_base.yaml: {bad}", problems=bad)


# ── the pair's inputs (level 0's functions for the trio pair, imported unchanged) ──


@dataclass
class Opened:
    inputs: dict
    contexts: dict
    handles: dict
    pkg: tuple
    bank: object
    m3b_compile: object


def pair_verify(pair: Pair) -> None:
    if pair.name == "trio":
        L0.verify_pins(L0.load_declaration())
    else:
        verify_six(SB.load_declaration())


def pair_open(pair: Pair, cfg: dict, cfg_m3b: dict, name: str) -> Opened:
    """The trio pair opens its one dataset, as level 0 did. The six pair opens the six through universal_v2_six.open_six,
    as stage 2's eval pass did, so the served relation bank and every offset are the ones its GNN was scored with."""
    if pair.name == "trio":
        inputs = V2.model_inputs(cfg, cfg_m3b)
        m3b_compile = V2.M3B_RUN.load_script("m3b_compile")
        contexts, handles, pkg, bank = V2.open_contexts_v2(cfg_m3b, [name], m3b_compile)
    else:
        inputs, _freeze, contexts, handles, pkg, bank, _carves, m3b_compile = U6.open_six(cfg, cfg_m3b, kinds=())
    return Opened(inputs, contexts, handles, pkg, bank, m3b_compile)


def checkpoint_problems(d: Path, key: str, arm: str) -> list[str]:
    """A checkpoint is its fit record's arm and its weights are the record's state_sha256."""
    rec = V2.read_json(d / f"{key}.json")
    if rec is None or not (d / f"{key}.pt").exists():
        return [f"{key}: no fit record or weights under {d}"]
    bad = []
    if rec.get("arm") != arm:
        bad.append(f"{key}: fit record arm {rec.get('arm')} is not {arm}")
    if SB.sha256_file(d / f"{key}.pt") != rec.get("state_sha256"):
        bad.append(f"{key}: the weights are not the state_sha256 of their record")
    return bad


def pair_models(pair: Pair, inputs: dict, bank) -> dict:
    if pair.name == "trio":
        decl0 = L0.load_declaration()
        selection = V2.read_json(ROOT / decl0["inputs"]["selection"]["path"])
        return L0.load_models(decl0, V2, inputs, bank, selection)
    models = {}
    for k in SEEDS:
        for fam, key, d, arm, params in (("twin", SB.fit_key(k), FITS, SB.ARM, SB.PARAMETERS),
                                         ("gnn", U6.fit_key(k), U6.SIX_FITS, U6.ARM, U6.PARAMETERS)):
            bad = checkpoint_problems(d, key, arm)
            if bad:
                hard_stop(f"{key}: {bad}", problems=bad)
            m = V2.make_model(arm, inputs, bank)
            if SB.parameter_count(m) != params:
                hard_stop(f"{key}: {SB.parameter_count(m)} parameters, not {params}")
            m.load_state_dict(torch.load(d / f"{key}.pt", map_location="cpu"))
            m.eval()
            models[f"{fam}{k}"] = m
    return models


def six_stored(name: str, d: Path = STAGE2_EVAL) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """Stage 2's per-query metrics of G_k over its whole eval population, in population order, with its halves (the
    trio only; True = V2_GATE) and hops (metaqa's from the ids, 0 elsewhere)."""
    with np.load(d / f"{name}.npz") as z:
        half = z["half"].astype(bool) if "half" in z.files else None
        hop = z["hop"].astype(np.int64)
        out = {f"gnn{k}": np.stack([z[f"{U6.fit_key(k)}/{m}"] for m in METRIC_NAMES], 1).astype(np.float64) for k in SEEDS}
    return half, hop, out


def pair_stored(pair: Pair, name: str):
    return L0.load_stored(L0.load_declaration(), name) if pair.name == "trio" else six_stored(name)


def pair_declared(pair: Pair, cfg: dict, name: str) -> dict:
    if pair.name == "trio":
        return cfg["m3b_incumbents"]["eval_populations_reused_here"][name]
    return U6.declared_population(U6.stage2_block(cfg), name)


def pair_ids_file(pair: Pair, name: str) -> Path:
    if pair.name == "trio":
        return ROOT / L0.load_declaration()["inputs"]["eval_arrays"][name]["query_ids"]["path"]
    return STAGE2_EVAL / f"{name}_query_ids.json"


def six_rows(name: str, ids: list[str], half: np.ndarray | None) -> np.ndarray:
    """work_2.populations: level 0's rows on the trio (metaqa's 2,400 of V2_GATE, every V2_GATE row of 2wiki and squad),
    every query of stage 2's population on hotpotqa, musique and webqsp."""
    if name in TRIO:
        return L0.scored_rows(name, ids, half)
    return np.arange(len(ids), dtype=np.int64)


def pair_rows(pair: Pair, name: str, ids: list[str], half: np.ndarray | None) -> np.ndarray:
    return L0.scored_rows(name, ids, half) if pair.name == "trio" else six_rows(name, ids, half)


# ── the copied scoring pass ──────────────────────────────────────────────────


def score_pair(pair: Pair, name: str, log=print, limit: int | None = None, out_dir: Path | None = None,
               shard: tuple[int, int] | None = None) -> None:
    """Level 0's stage_score (scripts/mp_approx_l0.py) copied under a new name with the pair passed in: its checkpoints,
    the functions with stored metrics, the population's declaration, id list, halves and rows, the pins and the thread
    count. Every computation, its order and every array written are level 0's. With a shard (amendment 2) the process
    scores only its chunks and files shard_<i>of<n>.json; a later run without one assembles."""
    torch.set_num_threads(pair.threads)
    out_dir = out_dir or OUT / name
    if (out_dir / "meta.json").exists():
        log(f"{out_dir}/meta.json exists; not rescored")
        return
    pair_verify(pair)
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    split = cfg_m3b["populations"]["eval_splits"][name]
    if "test" in str(split).lower():
        hard_stop(f"{name}: eval split {split} is a test split")
    op = pair_open(pair, cfg, cfg_m3b, name)
    inputs, m3b_compile = op.inputs, op.m3b_compile
    n_sc = int(inputs["n_scalars"])
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    context, ds = op.contexts[name], op.handles[name]
    models = pair_models(pair, inputs, op.bank)
    for k in SEEDS:
        twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
        if twin.input.input_width != 2 * n_sc + L0.N_VECTOR or gnn.steps != L0.STEPS or gnn.message_passing is not True:
            hard_stop(f"seed {k}: not the declared architecture (input width {twin.input.input_width}, steps {gnn.steps})")
    half_stored, hop_stored, stored = pair_stored(pair, name)
    m3a = op.pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    declared = pair_declared(pair, cfg, name)
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]):
        hard_stop(f"{name}: not the declared eval population", digest=pop.digest, queries=int(pop.idx.size))
    if pair.name == "six" and (declared["split"] != split or declared["pool"] != frozen["per_dataset"][name]["pool"]
                               or int(declared.get("zero_gold_excluded", pop.zero_gold_excluded)) != pop.zero_gold_excluded):
        hard_stop(f"{name}: the split, pool or zero-gold exclusion is not stage 2's", split=split, pool=frozen["per_dataset"][name]["pool"],
                  zero_gold_excluded=int(pop.zero_gold_excluded))
    ids_file = json.loads(pair_ids_file(pair, name).read_text(encoding="utf-8"))
    if list(pop.ids) != ids_file:
        hard_stop(f"{name}: the population ids are not the stored id list")
    half = None
    if half_stored is not None:
        half = V2.half_labels(name, ds, split, pop.ids)
        if not np.array_equal(half, half_stored):
            hard_stop(f"{name}: the recomputed halves differ from the stored halves")
    if name == "metaqa" and not np.array_equal(np.asarray([L0.hop_from_id(q) for q in pop.ids]), hop_stored):
        hard_stop("metaqa: the hops read from the ids differ from the stored hops")
    rows = pair_rows(pair, name, pop.ids, half)
    if limit is not None:
        rows = rows[:limit]
    if half is not None and not half[rows].all():
        hard_stop(f"{name}: a held row would be scored")
    all_ids = list(pop.ids)
    pop.ids, pop.idx, pop.golds = [all_ids[i] for i in rows], pop.idx[rows], [pop.golds[i] for i in rows]
    t_prep = time.time()
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    n = len(rows)
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(L0.CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    scope = "all" if half is None else f"{int(half.sum())} V2_GATE"
    log(f"{name}: {n} queries scored ({scope} of {len(all_ids)}), pools mean {sizes.mean():.0f}, chunk {chunk} queries, "
        f"{n_chunks} chunks, prepared in {time.time() - t_prep:.0f}s; pair {pair.name}, {torch.get_num_threads()} threads")
    columns = inputs["column_indices"]
    taps = {k: L0.Taps(models[f"twin{k}"], models[f"gnn{k}"], 2 * n_sc) for k in SEEDS}
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    mine = shard_chunks(n_chunks, shard)
    t0, done_here = time.time(), 0
    with torch.no_grad():
        for pos, ci in enumerate(mine):
            idx = np.arange(ci * chunk, min((ci + 1) * chunk, n))
            path = chunks_dir / f"c{ci:05d}.npz"
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z["chunk_rows"], rows[idx]):
                        raise SystemExit(f"{path}: not this chunk's rows; delete {chunks_dir} to rescore")
                continue
            qds, gold_locals, xs, firsts = [], [], [], []
            for j in idx:
                E = context.nodes.read(prep.pools[j])
                inp = V2.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = V2.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
                gl = m3b_compile.gold_local_of(prep.pools[j], pop.golds[j])
                x = compiled.scalars[:, columns]
                qds.append({"pool": compiled.pool, "x": x, "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(pop.golds[j].size), "emb": E})
                gold_locals.append(gl)
                xs.append(np.asarray(x, dtype=np.float32))
                firsts.append(L0.first_support(compiled.scalars, gl, V2.IDX))
                del compiled
            batch = V2.pack_queries_v2(qds, context)
            del qds
            ptr = batch.qptr.numpy()
            scores = {}
            for k in SEEDS:
                twin, gnn, tap = models[f"twin{k}"], models[f"gnn{k}"], taps[k]
                tap.reset()
                tap.on = True
                scores[f"twin{k}"] = twin(V2.arm_view(twin, batch, inputs)).numpy().astype(np.float64)
                scores[f"gnn{k}"] = gnn(V2.arm_view(gnn, batch, inputs)).numpy().astype(np.float64)
                tap.on = False
                if len(tap.copies) != L0.STEPS or len(tap.actual) != L0.STEPS:
                    hard_stop(f"seed {k}, chunk {ci}: {len(tap.copies)} copied and {len(tap.actual)} cell messages, not {L0.STEPS}")
                for t in range(L0.STEPS):
                    try:
                        torch.testing.assert_close(tap.copies[t], tap.actual[t], atol=1e-5, rtol=1e-5)
                    except AssertionError as err:
                        hard_stop(f"message_split: seed {k}, chunk {ci}, step {t + 1}: the copied attention is not the cell's message",
                                  detail=str(err)[:2000])
                scores[f"noedge{k}"] = L0.noedge_forward(gnn, V2.arm_view(gnn, batch, inputs)).numpy().astype(np.float64)
            cap = {k: {"channels": taps[k].channels.numpy(), "body": taps[k].body.numpy(), "base_z": taps[k].base_z.numpy(),
                       "nbr": [m.numpy() for m in taps[k].nbr]} for k in SEEDS}
            per_row = {key: [] for key in ("query", "local", "is_gold", "x", "zx", "z", "rank")}
            for k in SEEDS:
                per_row.update({f"channels_{k}": [], f"body_{k}": [], f"base_z_{k}": [], f"mnbr_{k}": []})
            per_q = {key: [] for key in ("q_row", "q_hop", "q_gold_total", "q_pool_size", "q_uq_size", "q_fold", "q_first_support_STRUCT", "q_metrics")}
            for jj, j in enumerate(idx):
                a, b = int(ptr[jj]), int(ptr[jj + 1])
                if b - a != sizes[j]:
                    hard_stop(f"query {pop.ids[j]}: packed rows {b - a} != pool size {sizes[j]}")
                gl, gt, row = gold_locals[jj], int(pop.golds[j].size), int(rows[j])
                full = {f: rank_metrics(scores[f][a:b], gl, gt) for f in L0.FUNCS}
                for f in pair.stored:
                    want = stored[f][row]
                    for mi, m in enumerate(METRIC_NAMES):
                        if full[f][m] != want[mi]:
                            hard_stop(f"integrity.stored: query {pop.ids[j]} (row {row}), {f}, {m}: forward {full[f][m]} != stored {want[mi]}",
                                      query=pop.ids[j], row=row, function=f, metric=m, forward=full[f][m], stored=float(want[mi]))
                loc = L0.analysis_set([scores[f][a:b] for f in L0.FUNCS], gl)
                gl_u = np.searchsorted(loc, gl)
                for f in L0.FUNCS:
                    r_u = rank_metrics(scores[f][a:b][loc], gl_u, gt)
                    for m in L0.UQ_EXACT:
                        if r_u[m] != full[f][m]:
                            hard_stop(f"integrity.analysis_set: query {pop.ids[j]} (row {row}), {f}, {m}: U_q {r_u[m]} != full pool {full[f][m]}",
                                      query=pop.ids[j], row=row, function=f, metric=m, uq=r_u[m], full=full[f][m])
                per_row["query"].append(np.full(loc.size, j, dtype=np.int32))
                per_row["local"].append(loc.astype(np.int32))
                per_row["is_gold"].append(np.isin(loc, gl))
                per_row["x"].append(xs[jj][loc])
                per_row["zx"].append(L0.column_z(xs[jj])[loc].astype(np.float32))
                per_row["z"].append(np.stack([L0.pool_z(scores[f][a:b])[loc] for f in L0.FUNCS], 1))
                per_row["rank"].append(np.stack([L0.pool_rank(scores[f][a:b])[loc] for f in L0.FUNCS], 1).astype(np.int32))
                for k in SEEDS:
                    c = cap[k]
                    per_row[f"channels_{k}"].append(L0.to_f16(c["channels"][a:b][loc], f"channels_{k}"))
                    per_row[f"body_{k}"].append(L0.to_f16(c["body"][a:b][loc], f"body_{k}"))
                    per_row[f"base_z_{k}"].append(np.asarray(c["base_z"][a:b][loc], dtype=np.float32))
                    per_row[f"mnbr_{k}"].append(L0.to_f16(np.stack([c["nbr"][t][a:b][loc] for t in range(L0.STEPS)], 1), f"mnbr_{k}"))
                per_q["q_row"].append(row)
                per_q["q_hop"].append(int(hop_stored[row]))
                per_q["q_gold_total"].append(gt)
                per_q["q_pool_size"].append(b - a)
                per_q["q_uq_size"].append(int(loc.size))
                per_q["q_fold"].append(L0.fold_of(pop.ids[j]))
                per_q["q_first_support_STRUCT"].append(firsts[jj])
                per_q["q_metrics"].append([[full[f][m] for m in METRIC_NAMES] for f in L0.FUNCS])
            arrays = {key: np.concatenate(v) for key, v in per_row.items()}
            arrays.update({key: np.asarray(v, dtype=np.float64 if key == "q_metrics" else np.int64) for key, v in per_q.items()})
            arrays["chunk_rows"] = rows[idx]
            tmp = chunks_dir / f"c{ci:05d}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            del batch, scores, cap, arrays, per_row, per_q, xs
            for tap in taps.values():
                tap.reset()
            gc.collect()
            done_here += idx.size
            if pos % max(1, len(mine) // 25) == 0 or pos == len(mine) - 1:
                rate = (time.time() - t0) / done_here
                left = sum(min((cj + 1) * chunk, n) - cj * chunk for cj in mine[pos + 1:])
                log(f"   {name}: chunk {ci + 1}/{n_chunks} ({pos + 1}/{len(mine)} of this process), {rate * 1000:.0f} ms/query, "
                    f"about {left * rate / 60:.0f} min left; integrity equal so far")
    for tap in taps.values():
        tap.remove()
    pair_verify(pair)   # again at the end
    if shard is not None:
        rec = {"dataset": name, "pair": pair.name, "shard": list(shard), "chunks": mine, "n_chunks": n_chunks, "chunk_queries": chunk,
               "queries_scored_here": done_here, "utc": L0.utc(), "seconds_this_process": round(time.time() - t0, 1),
               "threads": torch.get_num_threads(), "blas_threads": blas_env(), "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(),
               "placement": dict(PLACEMENT), "module_sha256": SB.module_shas(), "git_head": L0.git_head()}
        write_json(out_dir / f"shard_{shard[0]}of{shard[1]}.json", rec)
        log(f"{name}: shard {shard[0]}/{shard[1]} wrote {len(mine)} chunks ({done_here} queries here); a run without --shard assembles")
        return
    meta = L0.assemble(chunks_dir, out_dir, n_chunks)
    (out_dir / "qids.json").write_text(json.dumps(list(pop.ids)), encoding="utf-8")
    meta.update({"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": SB.lf_sha256(CONFIG),
                 "pair": pair.name, "keys": pair.keys(), "stored_functions": list(pair.stored),
                 "queries": n, "population_rows_scored": "q_row", "population_queries": len(all_ids), "limit": limit,
                 "chunk_queries": chunk, "chunks": n_chunks, "threads": torch.get_num_threads(), "blas_threads": blas_env(),
                 "seconds_this_process": round(time.time() - t0, 1), "functions": list(L0.FUNCS), "metric_names": list(METRIC_NAMES),
                 "columns": list(inputs["columns"]), "keep_top": L0.KEEP_TOP, "steps": L0.STEPS, "vector_channels": L0.N_VECTOR,
                 "mismatches": 0, "integrity": INTEGRITY[pair.name], "peak_rss_bytes": V2.M3B_RUN.peak_rss_bytes(),
                 "module_sha256": SB.module_shas(), "qids_sha256": SB.sha256_file(out_dir / "qids.json"),
                 "placement": dict(PLACEMENT), "queries_scored_here": done_here,
                 "shards": {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("shard_*.json"))}})
    write_json(out_dir / "meta.json", meta)
    shutil.rmtree(chunks_dir)
    log(f"{name}: scored {n} queries, {meta['uq_rows']} U_q rows, 0 mismatches")


# ── stage: equivalence (the copy against level 0 on level 0's trio pair) ─────

COMPARED = ("arrays_sha256", "arrays_shape", "uq_rows", "queries", "chunk_queries", "chunks", "qids_sha256", "functions",
            "metric_names", "columns", "keep_top", "steps", "vector_channels", "mismatches")


def compare_sidecars(a: dict, b: dict) -> list[str]:
    """The meta.json fields that must be equal for the copy to be level 0's: every array's sha256 and shape, the id
    list's sha256, the chunking and the layout."""
    return [k for k in COMPARED if a.get(k) != b.get(k)]


def stage_equivalence(name: str, log=print, limit: int | None = None, root: Path = EQUIV) -> dict:
    """work_2.path: level 0's stage_score and the copy (score_pair), both on level 0's trio pair, on the first `limit` of
    level 0's rows of one trio dataset, one after the other in this process at level 0's threads. Equal when every
    array is byte-identical; a difference is a hard stop and both sidecars are kept for the evidence."""
    if name not in TRIO:
        raise SystemExit(f"{name}: the equivalence runs on level 0's trio {list(TRIO)}")
    limit = EQUIVALENCE_LIMITS[name] if limit is None else int(limit)
    rec_path = root / f"{name}.json"
    if rec_path.exists():
        log(f"{rec_path} exists; not repeated")
        return json.loads(rec_path.read_text(encoding="utf-8"))
    a_dir, b_dir = root / "level0" / name, root / "copy" / name
    for d in (a_dir, b_dir):
        if d.exists():
            shutil.rmtree(d)   # this stage's own scratch from an interrupted run
    before = L0.HARD_STOP_DIR[0]
    L0.HARD_STOP_DIR[0] = root / "level0"   # level 0's smoke-check convention: its own directory, never outputs/mp_approx_l0
    try:
        t = time.time()
        L0.stage_score(L0.load_declaration(), name, log, limit=limit, out_dir=a_dir)
        s_level0 = round(time.time() - t, 1)
    finally:
        L0.HARD_STOP_DIR[0] = before
    t = time.time()
    score_pair(TRIO_PAIR, name, log, limit=limit, out_dir=b_dir)
    s_copy = round(time.time() - t, 1)
    a = json.loads((a_dir / "meta.json").read_text(encoding="utf-8"))
    b = json.loads((b_dir / "meta.json").read_text(encoding="utf-8"))
    differ = compare_sidecars(a, b)
    rec = {"dataset": name, "utc": L0.utc(), "pair": "trio (level 0's T_k and G_k)", "limit": limit, "equal": not differ,
           "differ": differ, "compared": list(COMPARED), "arrays_sha256": a["arrays_sha256"],
           "copy_arrays_sha256": b["arrays_sha256"], "queries": a["queries"], "uq_rows": a["uq_rows"], "chunks": a["chunks"],
           "chunk_queries": a["chunk_queries"], "threads": {"level0": a["threads"], "copy": b["threads"]},
           "blas_threads": blas_env(), "seconds": {"level0": s_level0, "copy": s_copy}, "placement": dict(PLACEMENT),
           "level_0_lf_sha256": SB.lf_sha256(ROOT / LEVEL_0_REL), "module_sha256": SB.module_shas(), "git_head": L0.git_head()}
    write_json(rec_path, rec)
    if differ:
        hard_stop(f"equivalence: the copy is not level 0's on {name}: {differ} differ", dataset=name, differ=differ)
    shutil.rmtree(a_dir)
    shutil.rmtree(b_dir)
    log(f"equivalence {name}: {len(a['arrays_sha256'])} arrays byte-identical over {a['queries']} queries in {a['chunks']} chunks")
    return rec


# ── stage: fits-record (work 1's record) ─────────────────────────────────────


def fits_record(fits_dir: Path = FITS) -> dict:
    """execution: fits_record_<date>, a transcription of the three fit records once all three exist. The select numbers
    are the early-stopping trace of each fit and are never read as a result (role)."""
    filed = SB.filed_fits(fits_dir)
    if sorted(filed) != list(SEEDS):
        raise SystemExit(f"fit records of seeds {sorted(filed)} exist; fits_record follows all three")
    fits, seconds = {}, 0.0
    for k in SEEDS:
        key, r = SB.fit_key(k), filed[k]
        if (fits_dir / f"{key}.ckpt").exists():
            raise SystemExit(f"{key}: a checkpoint is still present; the fit is not finished")
        if SB.sha256_file(fits_dir / f"{key}.pt") != r["state_sha256"]:
            hard_stop(f"{key}: the weights are not the state_sha256 of their record")
        if r["arm"] != SB.ARM or int(r["parameters"]) != SB.PARAMETERS or int(r["seed"]) != k:
            hard_stop(f"{key}: the record's arm, parameters or seed are not the declared ones")
        fits[key] = {"seed": k, "epochs_run": int(r["epochs_run"]), "best_epoch": int(r["best_epoch"]),
                     "best_epoch_at_the_cap": int(r["best_epoch"]) == MAX_EPOCHS - 1, "seconds": round(float(r["seconds"]), 1),
                     "epoch_seconds": [h["seconds"] for h in r["history"]],
                     "select_macro_recall5_trace": [round(float(h["select_macro_recall@5"]), 4) for h in r["history"]],
                     "state_sha256": r["state_sha256"], "record_sha256": SB.sha256_file(fits_dir / f"{key}.json"),
                     "threads": r["threads"], "blas_threads": r["blas_threads"], "pack_workers": r["pack_workers"],
                     "peak_rss_bytes": r["peak_rss_bytes"], "placement": r["placement"], "git_head": r["git_head"],
                     "script_lf_sha256": r["module_sha256"].get("scripts/mp_approx_six_base.py"), "utc": r["utc"]}
        seconds += float(r["seconds"])
    scripts = {v["script_lf_sha256"] for v in fits.values()}
    return {"work": "work_1_twin_fits", "arm": SB.ARM, "parameters": SB.PARAMETERS, "fits": fits,
            "fit_hours": round(seconds / 3600, 3), "ceiling_fit_hours": SB.CEILING_HOURS,
            "within_ceiling": seconds / 3600 <= SB.CEILING_HOURS, "one_script": len(scripts) == 1,
            "select_numbers": "the early-stopping trace of each fit only; never read as a result (role)",
            "next": "work 2 (its code is scripts/mp_approx_six_base_score.py); then run_record_<date>"}


def append_block(key: str, block: dict, config: Path = CONFIG, status_to: str | None = None) -> None:
    """A dated block appended to the declaration (outputs.record), read back identically; the status line moves only
    when status_to is given."""
    text = config.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    if key in decl:
        raise SystemExit(f"{key} exists")
    if status_to is not None:
        if decl["status"] != "DECLARED_NOT_RUN":
            raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
        text = text.replace("status: DECLARED_NOT_RUN", f"status: {status_to}", 1)
    dumped = yaml.safe_dump(L0.clean({key: block}), sort_keys=False, width=160, allow_unicode=True)
    config.write_text(text.rstrip(LF) + LF + LF + dumped, encoding="utf-8")
    back = yaml.safe_load(config.read_text(encoding="utf-8"))[key]
    if back != L0.clean(block):
        raise SystemExit(f"{key}: the appended block does not read back identically")


def stage_fits_record(date: str, log=print) -> None:
    block = fits_record()
    block = {"filed": date, **block}
    append_block(f"fits_record_{date}", block)
    log(f"filed fits_record_{date}: {block['fit_hours']:.2f} fit-hours, best epochs "
        f"{[v['best_epoch'] for v in block['fits'].values()]}")


# ── stage: score, run, status ────────────────────────────────────────────────


def stage_score(name: str, log=print, shard: tuple[int, int] | None = None) -> None:
    if name not in DATASETS:
        raise SystemExit(f"{name}: not a dataset of the base")
    score_pair(SIX, name, log, shard=shard)


def stage_verify(log=print, out: Path = OUT, datasets=DATASETS) -> dict:
    """amendment_2.sidecars: where the sidecars are, every array of each scored dataset re-read against its meta.json by
    level 0's Sidecar (which also holds the rows grouped by query and the fold rule), filed as verify_host.json. The file
    stage takes it for the arrays that stay on the host."""
    rec = {"utc": L0.utc(), "placement": dict(PLACEMENT), "module_sha256": SB.module_shas(), "git_head": L0.git_head(), "datasets": {}}
    for name in datasets:
        d = out / name
        if not (d / "meta.json").exists():
            raise SystemExit(f"{name}: not scored; verify follows the six scoring passes")
        sc = L0.Sidecar(d)
        rec["datasets"][name] = {"meta_sha256": SB.sha256_file(d / "meta.json"), "arrays_checked": len(sc.meta["arrays_sha256"]),
                                 "queries": sc.n_q, "uq_rows": sc.n_rows}
        log(f"verify {name}: {len(sc.meta['arrays_sha256'])} arrays equal their meta.json, {sc.n_q} queries, {sc.n_rows} U_q rows")
    write_json(out / HOST_VERIFY, rec)
    return rec


def filed_sidecar(name: str, d: Path, host_verified: dict | None):
    """The file stage's view of one sidecar: level 0's full check when every array is here; otherwise (amendment 2) the
    host's verify record must name this meta.json and its array count, each array fetched here must be its meta.json's,
    and the rows, ids and folds are checked by level 0's Sidecar without re-reading the arrays that stayed there."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if SB.sha256_file(d / "qids.json") != meta.get("qids_sha256"):
        hard_stop(f"{name}: qids.json is not the sha256 its meta.json records", dataset=name)
    files = meta["arrays_sha256"]
    here = sorted(f for f in files if (d / f).exists())
    if len(here) == len(files):
        return L0.Sidecar(d), "laptop"
    hv = ((host_verified or {}).get("datasets") or {}).get(name)
    if hv is None or hv.get("meta_sha256") != SB.sha256_file(d / "meta.json") or hv.get("arrays_checked") != len(files):
        hard_stop(f"{name}: the sidecar's arrays are not here and {HOST_VERIFY} does not vouch for this meta.json", dataset=name)
    lacking = [f for f in SIDECAR_FETCHED if f not in here]
    if lacking:
        raise SystemExit(f"{name}: fetch {lacking} from the host before filing")
    for f in here:
        if SB.sha256_file(d / f) != files[f]:
            hard_stop(f"{d / f}: not the sha256 its meta.json records", path=str(d / f))
    return L0.Sidecar(d, check=False), "host"


def run_step(args: list[str], log=print) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), *args]
    log(f"== {' '.join(args)}")
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"{' '.join(args)}: exited {rc}; the next step is not started")


def stage_run(log=print) -> None:
    """Work 2 in order, each step a fresh process: the equivalence on each trio dataset, then the six scoring passes, the
    cheap populations first. A step whose output exists is skipped; one at a time (placement.one_at_a_time)."""
    missing = preconditions(SB.load_declaration(), need_equivalence=False)
    if missing:
        raise SystemExit(f"work 2 does not start: {missing} not filed")
    for name in TRIO:
        if not (EQUIV / f"{name}.json").exists():
            run_step(["--stage", "equivalence", "--dataset", name], log)
    missing = preconditions(SB.load_declaration())
    if missing:
        raise SystemExit(f"work 2 does not score: {missing}")
    for name in RUN_ORDER:
        if not (OUT / name / "meta.json").exists():
            run_step(["--stage", "score", "--dataset", name], log)
    stage_status(log)


def stage_status(log=print) -> None:
    decl = SB.load_declaration()
    SB.stage_status(log)
    log(f"work 2 preconditions missing: {preconditions(decl) or 'none'}")
    for name in TRIO:
        rec = V2.read_json(EQUIV / f"{name}.json")
        log(f"equivalence {name}: {'not run' if rec is None else ('equal' if rec['equal'] else 'DIFFERS ' + str(rec['differ']))}")
    for name in DATASETS:
        meta = V2.read_json(OUT / name / "meta.json")
        chunks = sorted((OUT / name / "chunks").glob("c*.npz")) if (OUT / name / "chunks").exists() else []
        shards = sorted(p.stem for p in (OUT / name).glob("shard_*.json")) if (OUT / name).exists() else []
        log(f"{name}: " + (f"scored {meta['queries']} queries, {meta['uq_rows']} U_q rows" if meta else
                           f"{len(chunks)} chunks written" if chunks else "not scored") + (f"; shards filed {shards}" if shards else ""))
    rec = V2.read_json(OUT / HOST_VERIFY)
    if rec is not None:
        log(f"{HOST_VERIFY}: {sorted(rec['datasets'])} verified at {rec['utc']}")


# ── stage: file (run_record_<date>) ──────────────────────────────────────────


def descriptive_gap(q_metrics: np.ndarray) -> dict:
    """work_2.descriptive_gap: the seed-mean full-pool recall@5, hit@1 and full_coverage@5 of G_k and T_k over the scored
    queries, and G minus T. Filed as the gap the ladder's levels would approximate; no interval, no reading, never set
    beside a number of another file."""
    fi = {f: i for i, f in enumerate(L0.FUNCS)}
    mi = {m: i for i, m in enumerate(METRIC_NAMES)}
    out = {label: {m: float(np.mean([q_metrics[:, fi[f"{fam}{k}"], mi[m]].mean() for k in SEEDS])) for m in GAP_METRICS}
           for fam, label in (("gnn", "G"), ("twin", "T"))}
    out["G_minus_T"] = {m: out["G"][m] - out["T"][m] for m in GAP_METRICS}
    return {label: {m: round(v, 4) for m, v in vals.items()} for label, vals in out.items()}


def committed_lf_sha(commit: str, rel: str) -> str | None:
    try:
        data = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def held_rows_scored(name: str, q_row: np.ndarray) -> int:
    """How many scored rows are V2_HELD_CONFIRMATION rows (stage 2's stored halves); 0 on the datasets without halves."""
    half, _hop, _stored = six_stored(name)
    return 0 if half is None else int((~half[q_row]).sum())


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_<date> appended after the code check; status DECLARED_NOT_RUN -> RUN."""
    decl = SB.load_declaration()
    missing = preconditions(decl)
    if missing:
        raise SystemExit(f"not filed: {missing}")
    want = committed_lf_sha(commit, SCRIPT_REL)
    if want is None:
        raise SystemExit(f"{commit} does not hold {SCRIPT_REL}")
    ran = {}
    equivalence = {}
    for name in TRIO:
        rec = json.loads((EQUIV / f"{name}.json").read_text(encoding="utf-8"))
        ran[f"equivalence/{name}"] = rec["module_sha256"].get(SCRIPT_REL)
        equivalence[name] = {"equal": rec["equal"], "limit": rec["limit"], "queries": rec["queries"], "uq_rows": rec["uq_rows"],
                             "chunks": rec["chunks"], "arrays": len(rec["arrays_sha256"]), "threads": rec["threads"],
                             "placement": rec.get("placement", {"where": "laptop"}),
                             "record_sha256": SB.sha256_file(EQUIV / f"{name}.json")}
    host_verified = V2.read_json(OUT / HOST_VERIFY)
    per = {}
    for name in DATASETS:
        d = OUT / name
        sc, sidecar_at = filed_sidecar(name, d, host_verified)
        meta = sc.meta
        if meta["pair"] != "six" or meta["limit"] is not None or meta["mismatches"] != 0:
            hard_stop(f"{name}: not a full six-pair scoring pass", pair=meta["pair"], limit=meta["limit"])
        ran[f"score/{name}"] = meta["module_sha256"].get(SCRIPT_REL)
        shards = meta.get("shards") or {}
        for sname, srec in shards.items():
            ran[f"score/{name}/{sname}"] = srec["module_sha256"].get(SCRIPT_REL)
        q_row = np.load(d / "q_row.npy")
        held = held_rows_scored(name, q_row)
        if held:
            hard_stop(f"{name}: {held} held rows were scored")
        per[name] = {"queries": meta["queries"], "population_queries": meta["population_queries"], "uq_rows": meta["uq_rows"],
                     "chunks": meta["chunks"], "chunk_queries": meta["chunk_queries"], "seconds": meta["seconds_this_process"],
                     "threads": meta["threads"], "blas_threads": meta["blas_threads"], "peak_rss_bytes": meta["peak_rss_bytes"],
                     "mismatches": 0, "held_rows_scored": 0, "meta_sha256": SB.sha256_file(d / "meta.json"),
                     "qids_sha256": meta["qids_sha256"], "placement": meta.get("placement", {"where": "laptop"}),
                     "sidecar_at": sidecar_at,
                     "shards": {s: {"chunks": len(r["chunks"]), "queries_scored_here": r["queries_scored_here"],
                                    "seconds": r["seconds_this_process"], "peak_rss_bytes": r["peak_rss_bytes"],
                                    "threads": r["threads"]} for s, r in shards.items()},
                     "queries_scored_by_the_assembling_process": meta.get("queries_scored_here"),
                     "descriptive_gap": descriptive_gap(np.load(d / "q_metrics.npy"))}
    if any(v["sidecar_at"] == "host" for v in per.values()):
        ran["verify_host"] = host_verified["module_sha256"].get(SCRIPT_REL)
    bad = sorted(k for k, v in ran.items() if v != want)
    if bad:
        hard_stop(f"identical_code: {bad} did not run the committed {SCRIPT_REL}", problems=bad)
    work_2_where = sorted({v["placement"].get("where", "laptop") for v in per.values()}
                          | {v["placement"].get("where", "laptop") for v in equivalence.values()})
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "code_commit": commit,
           "code": SCRIPT_REL, "placement": (f"work 1 on the laptop CPU at 8 threads; work 2 on the {' and '.join(work_2_where)} CPU "
                                             f"(amendment 2 where host), the equivalence at level 0's 6 threads, the scoring passes at 4"),
           "held_rows_read": False, "test_rows_read": False, "checkpoints_updated": 0,
           "fits_record": max(k for k in decl if k.startswith("fits_record_")), "equivalence": equivalence,
           "datasets": per, "descriptive_gap_note": "work_2.descriptive_gap: filed, not a result; no interval, no reading, never "
                                                     "set beside a number of another file"}
    if host_verified is not None:
        run["verify_host"] = {"utc": host_verified["utc"], "placement": host_verified["placement"],
                              "record_sha256": SB.sha256_file(OUT / HOST_VERIFY), "datasets": sorted(host_verified["datasets"])}
    if extra:
        run.update(extra)
    append_block(f"run_record_{date}", run, status_to="RUN")
    log(f"filed run_record_{date}; status RUN")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("fits-record", "equivalence", "score", "run", "status", "file", "verify"))
    ap.add_argument("--dataset", help="equivalence and score: one dataset per process")
    ap.add_argument("--date", help="fits-record and file: the record's date, e.g. 2026_10_01")
    ap.add_argument("--commit", help="file: the commit every work-2 process ran from")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    ap.add_argument("--host", action="store_true", help="equivalence, score, verify, status: on the host, the mirror in place of the package (amendment 2)")
    ap.add_argument("--shard", default=None, help="score: i/n, the chunks ci with ci mod n = i (amendment 2); a later run without it assembles")
    args = ap.parse_args(argv)
    if args.stage in ("equivalence", "score") and args.dataset is None:
        ap.error(f"--stage {args.stage} needs --dataset")
    if args.stage in ("fits-record", "file") and not args.date:
        ap.error(f"--stage {args.stage} needs --date")
    if args.host and args.stage in ("fits-record", "file", "run"):
        ap.error(f"--stage {args.stage} runs on the laptop (amendment 2)")
    if args.shard is not None and args.stage != "score":
        ap.error("--shard is for --stage score")
    shard = parse_shard(args.shard)
    if args.stage == "verify" and not args.host:
        ap.error("--stage verify runs where the sidecars are, on the host (--host)")
    if args.host:
        host_mode(SB.load_declaration(), log_utc)
    if args.stage == "fits-record":
        stage_fits_record(args.date, log_utc)
    elif args.stage == "equivalence":
        stage_equivalence(args.dataset, log_utc)
    elif args.stage == "score":
        stage_score(args.dataset, log_utc, shard)
    elif args.stage == "verify":
        stage_verify(log_utc)
    elif args.stage == "run":
        stage_run(log_utc)
    elif args.stage == "status":
        stage_status(log_utc)
    else:
        if not args.commit:
            ap.error("--stage file needs --commit")
        extra = json.loads(args.extra.read_text(encoding="utf-8")) if args.extra else None
        stage_file(args.date, args.commit, log_utc, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
