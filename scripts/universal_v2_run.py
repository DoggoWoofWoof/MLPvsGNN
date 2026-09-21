"""Universal-v2 run tooling (configs/universal_v2.yaml): the pilot stages in
their declared order, each refusing what the declaration forbids.

    python scripts/universal_v2_run.py --stage compile [--datasets ..] [--kinds fit select]
    python scripts/universal_v2_run.py --stage screen                 # the fit caches only -> feature_screen.json
    python scripts/universal_v2_run.py --stage file --which contract --date YYYY_MM_DD
    python scripts/universal_v2_run.py --stage timing                 # one epoch of u_gnn_v2_ef, weights discarded
    python scripts/universal_v2_run.py --stage fit --arm u_gnn_v2 --seed 0
    python scripts/universal_v2_run.py --stage select                 # selection.json, written once
    python scripts/universal_v2_run.py --stage eval --datasets metaqa [--shard k/N]
    python scripts/universal_v2_run.py --stage merge --datasets metaqa
    python scripts/universal_v2_run.py --stage gate                   # V2_GATE only -> gate_record.json
    python scripts/universal_v2_run.py --stage file --which gate --date YYYY_MM_DD
    python scripts/universal_v2_run.py --stage eval --datasets metaqa --models u_gnn_v2_ef__H128__s1 u_gnn_v2_ef__H128__s2
                                                                  # seeds 1-2 after the gate: a supplement record beside the seed-0 record
    python scripts/universal_v2_run.py --stage fit --arm u_gnn_v2_ef --seed 1
                                                                  # amendment 4: seeds 1-2 of a selected arm that failed, under the filed replication
    python scripts/universal_v2_run.py --stage file --which replication --date YYYY_MM_DD

M3B is imported, never edited: the seven pinned files are hashed before any
stage runs (m3b_incumbents.pinned_files_sha256_lf_normalised). The M3B 78
core and its duplicate pairs are read from the committed block
qls_u_core_contract_2026_09_13 of the M3B declaration, never from a sidecar.
Sidecars live under outputs/universal_v2/ (gitignored). The CRAG package is
read-only (the M3B loader guard; a foreign byte-code cache is reported, never
removed). check_3_firewalls_made_explicit.refusals_in_code:
  - the screen reads outputs/universal_v2/cache/<dataset>/fit and refuses any other cache;
  - timing, fit, select, eval and gate need contract_frozen_<date> in the declaration;
  - eval needs selection.json; the selection is written once and refused after any eval record;
  - the gate needs selection.json, is read once, and never reads V2_HELD_CONFIRMATION;
  - the held column is produced by scripts/universal_v2_report.py, once.
"""

from __future__ import annotations

import os

# BLAS threads must be fixed before numpy loads; the work is parallelised across
# processes (one per dataset or shard), so each process keeps a small thread pool.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("M3B_BLAS_THREADS", "2"))

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import m3b_run as M3B_RUN  # noqa: E402  (pinned; imported, never edited)
from universal_v2_split_audit import in_v2_gate  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import FAMILIES, DenseNodes, QueryInputs  # noqa: E402
from mp_retrieval.m3b_models import parameter_count  # noqa: E402
from mp_retrieval.m3b_train import (METRIC_NAMES, CarveData, draw_indices, fit_model, listwise_loss, mrr_audit,  # noqa: E402
                                    pack_parts, rank_metrics)
from mp_retrieval.universal_v2_features import (BLOCK_OF, BLOCKS, CONTRACT_NAME, DEPTHS, IDX, MASK_COLUMNS, N_COLUMNS, N_DEPTH,  # noqa: E402
                                                 N_ORDERED, N_V2, ORDERED_COLUMNS, ORDERED_DEPTHS, V2_COLUMNS, CacheWriterV2,
                                                 compile_query_v2, contract_json_v2, screen_input_columns)
from mp_retrieval.universal_v2_models import (ARMS, FAMILY_OF_ARM, GNN_CANDIDATES, K_REL, N_EDGE_FEATURES_V2, PARAMETER_BUDGET,  # noqa: E402
                                               TWIN_CANDIDATES, CarveDataV2, UGNNv2, batch_view, build_arm, build_relation_bank,
                                               check_parameter_budget, pack_queries_v2, resolve_evidence_columns)

CONFIG = ROOT / "configs" / "universal_v2.yaml"
M3B_CONFIG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
HEADROOM_CONFIG = ROOT / "configs" / "m3a_headroom.yaml"
M3B_OUT = ROOT / "outputs" / "m3b"
OUT = ROOT / "outputs" / "universal_v2"
CACHE = OUT / "cache"
FITS = OUT / "fits"
EVAL = OUT / "eval"
DOC = ROOT / "docs" / "UNIVERSAL_V2_PILOT.md"     # written by scripts/universal_v2_report.py --stage doc
REPLICATION_DOC = ROOT / "docs" / "UNIVERSAL_V2_REPLICATION.md"     # --stage replication_doc (amendment 4); the pilot document is not re-rendered
PILOT = ("metaqa", "2wiki", "squad")
HALVES = ("V2_GATE", "V2_HELD_CONFIRMATION")
M3B_CORE_BLOCK = "qls_u_core_contract_2026_09_13"
M3B_CORE_SIZE = 78
HIDDEN, DROPOUT, HEADS, STEPS, D_R, D_P = 128, 0.2, 4, 3, 32, 8     # arms.shared and innovations 2-6; fixed, not screened
PACK = {"workers": 2, "depth": 2}    # batches packed ahead of the step (systems; --pack-workers / --prefetch-depth)
LATENCY_QUERIES = 500
FIXED_SCORERS = ("rrf", "support_h1_STRUCT", "support_h2_STRUCT", "support_h3_STRUCT", "qsupport_h1", "qsupport_h2", "qsupport_h3")
M3B_REFERENCES = {"gat_universal_v1": "gat_universal_v1__H128_L2__s0", "gat_no_mp_v1": "gat_no_mp_v1__H128_L2__s0",
                  "qls_u_sota_v1": "qls_u_sota_v1__H128__s0", "fixed_rrf": "fixed:rrf"}
BOOTSTRAP = {"resamples": 1000, "seed": 0, "level": 95}     # measurement.paired_procedures
DATED = re.compile("^(contract_frozen|timing|amendment_[0-9]+|pilot_gate_record|run_record|replication_record|hard_stop|authorization_stage_[0-9])_[0-9]{4}_[0-9]{2}_[0-9]{2}$")
LF = chr(10)
FAMILY_GATES = {"gnn": "GNN_GATE", "twin": "TWIN_GATE"}    # amendment 2 family_status_vocabulary
OVERALL = {(True, True): "BOTH_PASS", (True, False): "GNN_ONLY_PASS", (False, True): "TWIN_ONLY_PASS", (False, False): "PILOT_FAILED"}
FAMILY_FINAL = ("GATE_FAIL", "CONFIRMATION_FAIL", "CONFIRMED_PASS")           # amendment 3 terminal_state_vocabulary.per_family_final
TERMINAL = {(True, True): "BOTH_CONFIRMED", (True, False): "GNN_ONLY_CONFIRMED", (False, True): "TWIN_ONLY_CONFIRMED", (False, False): "PILOT_FAILED"}
HARD_STOP = "HARD_STOP_WITH_REASON"
FIT_HOURS_CEILING = 150.0                                     # compute.hard_ceiling; amendment 3 compute_ceiling_guard
REPLICATION_FAMILY = ("REPLICATION_PASS", "REPLICATION_FAIL")               # amendment 4 post_pilot_replication.status_vocabulary
REPLICATION_OVERALL = {(True, True): "BOTH_REPLICATION_PASS", (True, False): "GNN_REPLICATION_ONLY", (False, True): "TWIN_REPLICATION_ONLY",
                       (False, False): "BOTH_REPLICATION_FAIL"}


# ── the declaration, the pins, the M3B core ──────────────────────────────────


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def sha_of_names(names) -> str:
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()


def verify_pins(cfg: dict, root: Path = ROOT) -> dict:
    """m3b_incumbents.pinned_files_sha256_lf_normalised: every stage hashes the seven M3B files before it runs
    (compute.abort_criteria: any pin mismatch refuses to run)."""
    pins = cfg["m3b_incumbents"]["pinned_files_sha256_lf_normalised"]
    bad = sorted(rel for rel, want in pins.items() if not (root / rel).exists() or lf_sha256(root / rel) != want)
    if bad:
        raise SystemExit("M3B pin mismatch (m3b_incumbents.pinned_files_sha256_lf_normalised): " + ", ".join(bad) + "; nothing runs")
    return pins


def load_configs(config: Path = CONFIG, m3b_config: Path = M3B_CONFIG, headroom: Path = HEADROOM_CONFIG) -> tuple[dict, dict, dict]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg_m3b = yaml.safe_load(m3b_config.read_text(encoding="utf-8"))
    cfg_h = yaml.safe_load(headroom.read_text(encoding="utf-8"))
    verify_pins(cfg)
    return cfg, cfg_m3b, cfg_h


def m3b_core(cfg_m3b: dict) -> tuple[list[str], list[dict], str]:
    """The 78 M3B core names and the M3B screen's duplicate pairs, from the committed block only."""
    block = cfg_m3b[M3B_CORE_BLOCK]
    names = list(block["surviving"])
    sha = sha_of_names(names)
    if sha != block["sha256_of_comma_joined_surviving_names"] or len(names) != M3B_CORE_SIZE:
        raise SystemExit(f"{M3B_CORE_BLOCK}: the surviving list does not match its sha or its size")
    return names, list(block["duplicate_pairs"]), sha


def dated_blocks(cfg: dict, prefix: str) -> list[str]:
    return sorted(k for k in cfg if DATED.match(k) and k.startswith(prefix + "_"))


def frozen_contract_v2(cfg: dict) -> tuple[str, dict]:
    keys = dated_blocks(cfg, "contract_frozen")
    if not keys:
        raise SystemExit("no contract_frozen_<date> block in the declaration: the screen is filed before any later stage (check_3)")
    return keys[-1], cfg[keys[-1]]


def pinned_raw_contract(cfg: dict, core78: list[str]) -> tuple[str, dict]:
    """amendment 2 raw_contract_pinned: the exact ordered v2 column names and the screen-input count and sha, filed
    before any real cache; the code contract must equal it (correction 1: code == declaration == frozen)."""
    keys = [k for k in dated_blocks(cfg, "amendment") if isinstance(cfg[k], dict) and isinstance(cfg[k].get("raw_contract_pinned"), dict)]
    if not keys:
        raise SystemExit("no dated amendment pins the raw contract (raw_contract_pinned); the screen refuses to run on an unpinned list")
    key, pin = keys[-1], cfg[keys[-1]]["raw_contract_pinned"]
    names = screen_input_columns(core78)
    problems = []
    if list(pin["v2_column_names"]) != list(V2_COLUMNS):
        problems.append("v2_column_names differ from the code (universal_v2_features.V2_COLUMNS)")
    for field, value in (("m3b_core", M3B_CORE_SIZE), ("depth_basis_columns", N_DEPTH), ("ordered_relation_path_columns", N_ORDERED),
                         ("v2_columns", N_V2), ("screen_input_columns", len(names)), ("cache_layout_columns", N_COLUMNS)):
        if int(pin[field]) != value:
            problems.append(f"{field}: declaration {pin[field]} vs code {value}")
    if pin["v2_column_names_sha256"] != sha_of_names(V2_COLUMNS):
        problems.append("v2_column_names_sha256 differs from the code")
    if pin["screen_input_sha256"] != sha_of_names(names):
        problems.append("screen_input_sha256 differs from the code (the M3B 78 followed by the 86 v2 columns)")
    if problems:
        raise SystemExit(f"{key}.raw_contract_pinned does not match the code contract: " + "; ".join(problems) + "; nothing is compiled or screened")
    return key, pin


def hop_of(ids, dataset: str) -> np.ndarray:
    """measurement.slices_reported: metaqa by hop, read from the query id (metaqa:<k>hop:...) exactly as the
    eval stage slices it; 0 (one slice, all) elsewhere."""
    if dataset == "metaqa":
        return np.asarray([int(q.split(":")[1][0]) for q in ids], dtype=np.int64)
    return np.zeros(len(ids), dtype=np.int64)


def hop_label(h: int) -> str:
    return "all" if h == 0 else f"{h}hop"


class CompileDiagnostics:
    """amendment 2 / step 3 compile diagnostics over EVERY query of the carve, by hop (metaqa) or in one slice: the
    K_REL relation-slot truncation (structural pairs carrying more than K_REL stored relations, with the histogram of
    relations per pair) and the availability of the typed / ordered relation-path channel (candidate rows and gold rows
    with a typed STRUCT walk of length t, queries with any such row, best walks with an inverse step, best walks that
    compose two different relations)."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.acc: dict[int, dict] = {}
        self.walk_cols = {t: IDX[f"typed_walks_h{t}"] for t in DEPTHS}
        self.dir_cols = {t: [IDX[f"opath_h{t}_dir{k}"] for k in range(1, t + 1)] for t in ORDERED_DEPTHS}
        self.adj_cols = {t: [IDX[f"opath_h{t}_adj{k}{k + 1}"] for k in range(1, t)] for t in ORDERED_DEPTHS}

    def _slot(self, hop: int) -> dict:
        if hop not in self.acc:
            self.acc[hop] = {"queries": 0, "rows": 0, "gold_rows": 0, "pairs": 0, "truncated": 0, "entries": 0, "pairs_with": [],
                             "queries_with_truncation": 0, "typed_queries": 0,
                             **{f"rows_walk_h{t}": 0 for t in DEPTHS}, **{f"gold_rows_walk_h{t}": 0 for t in DEPTHS},
                             **{f"queries_walk_h{t}": 0 for t in DEPTHS}, **{f"queries_gold_walk_h{t}": 0 for t in DEPTHS},
                             **{f"rows_inverse_step_h{t}": 0 for t in ORDERED_DEPTHS}, **{f"rows_heterogeneous_h{t}": 0 for t in ORDERED_DEPTHS}}
        return self.acc[hop]

    def add(self, hop: int, X: np.ndarray, gold_local: np.ndarray, pair_counts: dict | None) -> None:
        a = self._slot(int(hop))
        a["queries"] += 1
        a["rows"] += int(X.shape[0])
        gold = np.asarray(gold_local, dtype=np.int64)
        a["gold_rows"] += int(gold.size)
        if pair_counts:
            a["typed_queries"] += 1
            a["pairs"] += pair_counts["pairs"]
            a["entries"] += pair_counts["entries"]
            hist = pair_counts["pairs_with"]
            if len(hist) > len(a["pairs_with"]):
                a["pairs_with"].extend([0] * (len(hist) - len(a["pairs_with"])))
            for k, c in enumerate(hist):
                a["pairs_with"][k] += int(c)
            truncated = int(sum(hist[K_REL:]))
            a["truncated"] += truncated
            a["queries_with_truncation"] += int(truncated > 0)
        for t in DEPTHS:
            walk = X[:, self.walk_cols[t]] > 0
            a[f"rows_walk_h{t}"] += int(walk.sum())
            a[f"queries_walk_h{t}"] += int(walk.any())
            gw = walk[gold] if gold.size else walk[:0]
            a[f"gold_rows_walk_h{t}"] += int(gw.sum())
            a[f"queries_gold_walk_h{t}"] += int(gw.any())
        for t in ORDERED_DEPTHS:
            has = X[:, self.walk_cols[t]] > 0
            a[f"rows_inverse_step_h{t}"] += int((X[:, self.dir_cols[t]] < 0).any(axis=1).sum())
            a[f"rows_heterogeneous_h{t}"] += int((has & (X[:, self.adj_cols[t]] < 1.0 - 1e-4).any(axis=1)).sum())

    @staticmethod
    def _merge(total: dict, a: dict) -> None:
        for k, v in a.items():
            if k == "pairs_with":
                if len(v) > len(total[k]):
                    total[k].extend([0] * (len(v) - len(total[k])))
                for i, c in enumerate(v):
                    total[k][i] += c
            else:
                total[k] += v

    def summary(self) -> dict:
        hops = sorted(self.acc)
        slots = {hop_label(h): self.acc[h] for h in hops}
        if len(hops) > 1:
            total = self._slot(-1)
            for h in hops:
                self._merge(total, self.acc[h])
            slots["all"] = self.acc.pop(-1)
        out = {"k_rel": K_REL, "by_hop": {}}
        for label, a in slots.items():
            q, rows, gold = max(a["queries"], 1), max(a["rows"], 1), max(a["gold_rows"], 1)
            out["by_hop"][label] = {
                "queries": a["queries"], "rows": a["rows"], "gold_rows": a["gold_rows"], "typed_queries": a["typed_queries"],
                "relation_slots": {"pairs": a["pairs"], "entries": a["entries"], "pairs_truncated": a["truncated"],
                                   "fraction_of_pairs_truncated": round(a["truncated"] / max(a["pairs"], 1), 6),
                                   "queries_with_any_truncated_pair": a["queries_with_truncation"],
                                   "fraction_of_queries_with_any_truncated_pair": round(a["queries_with_truncation"] / q, 6),
                                   "relations_per_pair_histogram": {str(k + 1): c for k, c in enumerate(a["pairs_with"]) if c},
                                   "max_relations_per_pair": len(a["pairs_with"])},
                "typed_walk_availability": {f"h{t}": {"rows": round(a[f"rows_walk_h{t}"] / rows, 6), "gold_rows": round(a[f"gold_rows_walk_h{t}"] / gold, 6),
                                                       "queries": round(a[f"queries_walk_h{t}"] / q, 6), "queries_gold": round(a[f"queries_gold_walk_h{t}"] / q, 6)}
                                            for t in DEPTHS},
                "ordered_path": {f"h{t}": {"rows_with_walk": a[f"rows_walk_h{t}"],
                                           "fraction_of_walks_with_an_inverse_step": round(a[f"rows_inverse_step_h{t}"] / max(a[f"rows_walk_h{t}"], 1), 6),
                                           "fraction_of_walks_composing_different_relations": round(a[f"rows_heterogeneous_h{t}"] / max(a[f"rows_walk_h{t}"], 1), 6)}
                                 for t in ORDERED_DEPTHS}}
        return out


def training_overrides(cfg: dict) -> list[tuple[str, dict]]:
    """Dated amendment blocks carrying a training_override (compute.timing_run's declared fallback), in key order."""
    return [(k, cfg[k]["training_override"]) for k in dated_blocks(cfg, "amendment")
            if isinstance(cfg[k], dict) and isinstance(cfg[k].get("training_override"), dict)]


def training_rule_v2(cfg: dict, cfg_m3b: dict) -> dict:
    """training.rule: the M3B training block as amended by M3B amendment 3 (the sampler reading is required),
    plus any v2 override filed as a dated amendment before the fits it applies to."""
    training = M3B_RUN.training_rule(cfg_m3b, require_reading=True)
    rule = {"max_epochs": int(training["max_epochs"]), "batches_per_epoch": int(str(training["epoch"]).split()[0]),
            "batch_size": int(training["batch_queries"]), "patience": 2, "lr": 1e-3, "weight_decay": 1e-4, "clip": 1.0,
            "dataset_draw": training["dataset_draw"], "epoch_limit_s": float(training.get("epoch_limit_s", 28800)),
            "reading_block": training.get("reading_block"), "override_block": None}
    for key, override in training_overrides(cfg):
        for k in ("max_epochs", "batches_per_epoch", "patience"):
            if k in override:
                rule[k] = int(override[k])
        rule["override_block"] = key
    return rule


def append_block(cfg: dict, key: str, block: dict, header: str, config: Path = CONFIG) -> None:
    """Append one dated block to the declaration, once per prefix; it must read back identically."""
    if not DATED.match(key):
        raise SystemExit(f"{key}: not a dated block key (order_of_operations)")
    prefix = key[:-11]
    if dated_blocks(cfg, prefix) or key in cfg:
        raise SystemExit(f"a {prefix}_* block is already filed; a change is a new dated amendment with its reason, never a re-file")
    text = yaml.safe_dump({key: block}, sort_keys=False, width=110, allow_unicode=True)
    with open(config, "a", encoding="utf-8", newline=LF) as f:
        f.write(LF + "# -- " + header + " --" + LF + text)
    reloaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    if reloaded.get(key) != block:
        raise SystemExit(f"{key}: the appended block does not read back identically")
    cfg[key] = block


def set_status(config: Path, new: str, allowed_from: tuple) -> str:
    """The one top-level status line (order_of_operations 6: PILOT_GATE_READ); the only edit above the dated blocks."""
    text = config.read_bytes().decode("utf-8")   # bytes: the line ending is kept as found (read_text would translate it)
    nl = chr(13) + LF if chr(13) + LF in text else LF
    lines = text.split(nl)
    hits = [i for i, line in enumerate(lines) if line.startswith("status: ")]
    if len(hits) != 1:
        raise SystemExit("the declaration must carry exactly one top-level status line")
    current = lines[hits[0]][len("status: "):].strip()
    if current not in allowed_from:
        raise SystemExit(f"status {current} does not move to {new}")
    lines[hits[0]] = "status: " + new
    with open(config, "w", encoding="utf-8", newline="") as f:
        f.write(nl.join(lines))
    return current


def model_inputs(cfg: dict, cfg_m3b: dict) -> dict:
    """UNIVERSAL_V2_CORE_CONTRACT as the arms read it (universal_v2_models.build_arm inputs): the frozen
    survivors, the base column, the evidence columns with their recorded substitutions, and the M3B 78 inside
    the same layout (the control and the ablation). Refuses before contract_frozen_<date>."""
    key, frozen = frozen_contract_v2(cfg)
    names = list(frozen["surviving"])
    if sha_of_names(names) != frozen["sha256_of_comma_joined_surviving_names"]:
        raise SystemExit(f"{key}: the surviving list does not match its sha")
    core78, pairs, sha78 = m3b_core(cfg_m3b)
    pin_key, _ = pinned_raw_contract(cfg, core78)
    if names[:M3B_CORE_SIZE] != core78:
        raise SystemExit(f"{key}: the first {M3B_CORE_SIZE} survivors are not the M3B core in its order")
    unknown = [n for n in names if n not in IDX]
    if unknown:
        raise SystemExit(f"{key}: columns outside the v2 layout: {unknown}")
    screened = screen_input_columns(core78)
    if frozen.get("raw_contract_sha256") != sha_of_names(screened) or int(frozen.get("screened_columns", -1)) != len(screened):
        raise SystemExit(f"{key}: the frozen block was screened from a raw contract other than the one the code and {pin_key} pin "
                         "(code == declaration == frozen is required)")
    if [n for n in screened if n not in names] != list(frozen.get("dropped", {})):
        raise SystemExit(f"{key}: the survivors are not the pinned raw contract minus the recorded drops, in order")
    evidence, subs = resolve_evidence_columns(names, pairs)
    evidence78, subs78 = resolve_evidence_columns(core78, pairs)
    return {"contract_block": key, "columns": names, "column_indices": np.asarray([IDX[n] for n in names], dtype=np.int64),
            "n_scalars": len(names), "core_sha256": frozen["sha256_of_comma_joined_surviving_names"], "base": "rrf",
            "base_local": names.index("rrf"), "evidence": evidence, "evidence_substitutions": subs,
            "evidence_local": [names.index(n) for n in evidence],
            "core78_columns": core78, "core78_indices": np.asarray([IDX[n] for n in core78], dtype=np.int64), "core78_sha256": sha78,
            "core78_base_local": core78.index("rrf"), "core78_evidence": evidence78, "core78_evidence_substitutions": subs78,
            "core78_evidence_local": [core78.index(n) for n in evidence78], "core78_positions": list(range(M3B_CORE_SIZE))}


# ── contexts, caches, the disk guard ─────────────────────────────────────────


def open_contexts_v2(cfg_m3b: dict, datasets: list[str], m3b_compile):
    """The M3B contexts (family stores, dense nodes, relation tables; scripts/m3b_run.open_contexts unchanged)
    and the served relation bank built over them (universal_v2_models.build_relation_bank)."""
    contexts, handles, pkg = M3B_RUN.open_contexts(cfg_m3b, datasets, m3b_compile)
    bank, contexts_v2 = build_relation_bank(contexts)
    return contexts_v2, handles, pkg, bank


def open_carves_v2(contexts: dict, inputs: dict, kinds=("fit", "select")) -> dict:
    """The v2 caches read three ways: CarveDataV2 over the frozen core (the v2 arms), CarveDataV2 over the M3B
    78 inside the same cache (u_gnn_v2_core78), and the pinned CarveData over the M3B 78 (gat_universal_v1_trio,
    packed 8 wide by the pinned pack_queries)."""
    out = {"v2": {k: {} for k in kinds}, "core78": {k: {} for k in kinds}, "control": {k: {} for k in kinds}}
    for name, ctx in contexts.items():
        for kind in kinds:
            d = CACHE / name / kind
            if not (d / "meta.json").exists():
                raise SystemExit(f"{name}/{kind}: no cache at {d}; run --stage compile first")
            out["v2"][kind][name] = CarveDataV2(d, ctx, columns=inputs["column_indices"])
            out["core78"][kind][name] = CarveDataV2(d, ctx, columns=inputs["core78_indices"])
            out["control"][kind][name] = CarveData(d, ctx, columns=inputs["core78_indices"])
    return out


def carves_for_arm(arm: str, carves: dict) -> dict:
    return carves["control"] if arm == "gat_universal_v1_trio" else carves["core78"] if arm == "u_gnn_v2_core78" else carves["v2"]


class DiskGuardV2:
    """compute.abort_criteria: halt below the declared free-disk floor; the cache under outputs/universal_v2/cache
    is bounded and recomputable (the carve being written is deleted at the bound)."""

    def __init__(self, cfg: dict):
        text = " ".join(str(line) for line in cfg["compute"]["abort_criteria"])
        floor = re.search("free disk below ([0-9]+) GB", text)
        bound = re.search("bounded at ([0-9]+) GB", text)
        if not floor or not bound:
            raise SystemExit("compute.abort_criteria does not declare the disk floor and the cache bound; refusing to write a cache")
        self.halt_below = float(floor.group(1)) * 1e9
        self.bound = float(bound.group(1)) * 1e9

    @staticmethod
    def cache_bytes() -> int:
        return sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file()) if CACHE.exists() else 0

    def check(self, writing: Path | None = None) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(OUT).free
        if free < self.halt_below:
            raise SystemExit(f"free disk {free / 1e9:.2f} GB below the {self.halt_below / 1e9:.0f} GB floor (compute.abort_criteria); halted")
        used = self.cache_bytes()
        if used >= self.bound:
            if writing is not None and writing.exists():
                shutil.rmtree(writing)
            raise SystemExit(f"cache {used / 1e9:.2f} GB reached its {self.bound / 1e9:.0f} GB bound (compute.abort_criteria); "
                             "the carve being written was deleted, compilation halted")


# ── compile: the fit and select carves under the v2 contract ─────────────────


def compile_population_v2(prep, stores: dict, nodes, rel_table, out_dir: Path, meta: dict, gold_local_of, log=print, guard=None) -> dict:
    """scripts/m3b_compile.compile_population under a v2 name: compile_query_v2 into a CacheWriterV2, and the
    step-3 diagnostics over every query (CompileDiagnostics: the K_REL relation-slot truncation by dataset and hop,
    the typed / ordered relation-path availability), the wall time, queries per second, peak RSS and bytes."""
    timings: dict = {}
    writer = CacheWriterV2(out_dir, meta)
    pop = prep.pop
    diag = CompileDiagnostics(meta["dataset"])
    hops = hop_of(pop.ids, meta["dataset"])
    t0 = time.time()
    for i in range(pop.idx.size):
        inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
        pair_counts: dict = {}
        compiled = compile_query_v2(inp, prep.pools[i], prep.seeds[i], stores, nodes, rel_table, timings=timings, pair_counts=pair_counts)
        gold_local = gold_local_of(prep.pools[i], pop.golds[i])
        writer.add(pop.ids[i], int(pop.idx[i]), prep.qemb[i], compiled, gold_local, int(pop.golds[i].size))
        diag.add(int(hops[i]), compiled.scalars, gold_local, pair_counts if rel_table is not None else None)
        if (i + 1) % 500 == 0:
            log(f"      {pop.kind}: {i + 1}/{pop.idx.size} queries, {(time.time() - t0) / (i + 1) * 1000:.1f} ms/query")
            if guard is not None:
                guard.check(writing=out_dir)
    seconds = time.time() - t0
    written = writer.write()
    n = max(1, pop.idx.size)
    diagnostics = diag.summary()
    overall = diagnostics["by_hop"]["all"]["relation_slots"]
    slots = {"k_rel": K_REL, "queries": int(pop.idx.size), "pairs": overall["pairs"], "truncated": overall["pairs_truncated"], "entries": overall["entries"],
             "fraction_of_pairs_truncated": overall["fraction_of_pairs_truncated"],
             "fraction_of_queries_with_any_truncated_pair": overall["fraction_of_queries_with_any_truncated_pair"]}
    written.update({"compile_seconds": round(seconds, 1), "ms_per_query": round(1000 * seconds / n, 2), "queries_per_second": round(n / max(seconds, 1e-9), 3),
                    "group_seconds_per_1000_queries": {g: round(1000 * s / n, 2) for g, s in timings.items()},
                    "expansion_seconds": round(prep.expansion_seconds, 1),
                    "seeds_added_mean": float(prep.seeds_added.mean()) if prep.seeds_added.size else 0.0,
                    "queries_with_seeds_added": int((prep.seeds_added > 0).sum()), "relation_slots": slots,
                    "peak_rss_bytes": M3B_RUN.peak_rss_bytes(), "diagnostics": diagnostics})
    (out_dir / "meta.json").write_text(json.dumps(written, indent=1), encoding="utf-8")
    return written


def stage_compile(cfg: dict, cfg_m3b: dict, cfg_h: dict, datasets: list[str], kinds: tuple[str, ...], log=print) -> None:
    """Order of operations 3: the fit and select carves of the pilot datasets under UNIVERSAL_V2_FEATURE_CONTRACT,
    the M3B carves by id (refused on any digest or size mismatch). The eval populations are never cached here:
    the eval stage compiles them per query, after contract_frozen_<date> exists (check_3)."""
    if any(k not in ("fit", "select") for k in kinds):
        raise SystemExit("compile caches the training carves only (fit, select); the eval populations are compiled at eval time")
    bad = [d for d in datasets if d not in PILOT]
    if bad:
        raise SystemExit(f"{bad}: not pilot datasets (training.pilot_datasets)")
    m3b_compile = M3B_RUN.load_script("m3b_compile")
    m3b_contract = M3B_RUN.load_script("m3b_contract")
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
    key_m3b, frozen = m3b_compile.frozen_contract(cfg_m3b)
    guard = DiskGuardV2(cfg)
    guard.check()
    core78, _, sha78 = m3b_core(cfg_m3b)
    pin_key, _ = pinned_raw_contract(cfg, core78)       # correction 1: the code contract equals the pinned declaration, or nothing compiles
    contract = contract_json_v2(core78)
    contract["m3b_core_sha256"] = sha78
    contract["raw_contract_pinned_in"] = pin_key
    contract["screen_input_sha256"] = sha_of_names(screen_input_columns(core78))
    (OUT / "feature_contract.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    log(f"   {CONTRACT_NAME}: {N_COLUMNS} columns in the cache layout; screen input {len(screen_input_columns(core78))} = {M3B_CORE_SIZE} + {N_V2} "
        f"({N_DEPTH} depth basis + {N_ORDERED} ordered relation path), pinned by {pin_key}")
    carves = cfg["m3b_incumbents"]["training_carves_reused_here"]
    for name in datasets:
        t_ds = time.time()
        construction = frozen["per_dataset"][name]["construction"]
        todo = []
        for kind in kinds:
            if (CACHE / name / kind / "meta.json").exists():
                log(f"   {name}/{kind}: cache exists, not repeated")
            else:
                todo.append(kind)
        if not todo:
            continue
        log(f"== {name}: {frozen['per_dataset'][name]['pool']} ({construction})")
        ds = canonical.Dataset(name, root=str(served))
        positions = m3a.node_position_map(ds)
        pops = [m3b_compile.population(ds, name, kind, cfg_m3b, cfg_h, m3a, positions) for kind in todo]
        del positions
        gc.collect()
        for p in pops:
            declared = carves[name]
            if p.n_before != int(declared[p.kind]) or (p.zero_gold_excluded == 0 and p.digest != declared[f"{p.kind}_sha256"]):
                raise SystemExit(f"{name}/{p.kind}: not the M3B carve (m3b_incumbents.training_carves_reused_here); refusing")
            log(f"   {p.kind}: {p.n_before} ids, {p.zero_gold_excluded} zero-gold excluded, {p.idx.size} kept (carve digest checked)")
        stores = {f: m3b_pools.load_or_build_store(ds, f, m3b_compile.CSR_CACHE) for f in FAMILIES}
        prepared = m3b_compile.prepare(ds, pops, construction, cfg_h, stores, m3a, m3b_contract)
        nodes = DenseNodes(ds.embeddings("dense", "docs"))
        rel_table = m3b_compile.relation_table_for(ds, name, stores)
        for prep in prepared:
            sizes = np.asarray([p.size for p in prep.pools])
            log(f"   {prep.pop.kind}: pools mean {sizes.mean():.0f} p95 {np.percentile(sizes, 95):.0f} max {sizes.max()}, "
                f"seeds added mean {prep.seeds_added.mean():.2f}, expansions {prep.expansion_seconds:.0f}s")
            meta = {"dataset": name, "kind": prep.pop.kind, "m3b_contract_block": key_m3b, "pool": frozen["per_dataset"][name]["pool"],
                    "construction": construction,
                    "population": {"ids": prep.pop.n_before, "zero_gold_excluded": prep.pop.zero_gold_excluded, "kept": int(prep.pop.idx.size),
                                   "ids_sha256": prep.pop.digest},
                    "carve_sha256_declared": carves[name][f"{prep.pop.kind}_sha256"], "feature_contract": CONTRACT_NAME, "n_columns": N_COLUMNS,
                    "m3b_core_sha256": sha78, "relation_table": rel_table is not None, "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "utc": utc()}
            guard.check()
            written = compile_population_v2(prep, stores, nodes, rel_table, CACHE / name / prep.pop.kind, meta, m3b_compile.gold_local_of,
                                            log=log, guard=guard)
            log(f"   {prep.pop.kind}: {written['n_queries']} queries, {written['n_rows']} rows, {written['ms_per_query']} ms/query, "
                f"{written['bytes'] / 1e9:.2f} GB, no-gold-in-pool {written['queries_with_no_gold_in_pool']}, "
                f"relation slots truncated {written['relation_slots']['fraction_of_pairs_truncated']}")
        del prepared, nodes, stores
        gc.collect()
        log(f"   {name}: {time.time() - t_ds:.0f}s")


# ── screen: the M3B rule on the fit caches only ──────────────────────────────


def screen_cache_dir(name: str) -> Path:
    """check_3: the screen reads outputs/universal_v2/cache/<dataset>/fit and refuses any other cache."""
    d = CACHE / name / "fit"
    if not (d / "meta.json").exists():
        raise SystemExit(f"{name}: no fit cache at {d}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("kind") != "fit" or meta.get("feature_contract") != CONTRACT_NAME or int(meta.get("n_columns", -1)) != N_COLUMNS:
        raise SystemExit(f"{d}: not a fit cache of {CONTRACT_NAME} (kind {meta.get('kind')}); the screen refuses any other cache")
    if meta.get("columns_stored") is not None:
        raise SystemExit(f"{d}: trimmed; the screen runs on the full layout")
    return d


def column_stats_v2(cache_dir: Path, take: np.ndarray, chunk_rows: int = 500_000) -> dict:
    """scripts/m3b_compile.column_stats under a v2 name, on the given columns of one fit cache."""
    scalars = np.load(cache_dir / "scalars.npy", mmap_mode="r")
    n = scalars.shape[0]
    nonzero, s1, s2 = np.zeros(take.size), np.zeros(take.size), np.zeros(take.size)
    for a in range(0, n, chunk_rows):
        X = np.asarray(scalars[a:a + chunk_rows], dtype=np.float64)[:, take]
        nonzero += (X != 0).sum(axis=0)
        s1 += X.sum(axis=0)
        s2 += (X * X).sum(axis=0)
    mean = s1 / max(n, 1)
    var = np.maximum(s2 / max(n, 1) - mean * mean, 0.0)
    return {"rows": int(n), "availability": nonzero / max(n, 1), "mean": mean, "variance": var}


def stage_screen(cfg: dict, cfg_m3b: dict, datasets: list[str], log=print) -> dict:
    """information_contract_v2.screen: the M3B feature_screen rule verbatim (availability, variance, |Spearman| >= 0.98
    with the later member dropped) on the 164 raw columns of the pilot fit caches (the M3B 78 + the 86 v2 columns
    pinned by amendment 2, checked against the code before anything is read), read from
    outputs/universal_v2/cache/<dataset>/fit only. The M3B 78 come first and always survive by construction (they
    are the earlier members and were screened by M3B; a re-screen here would let the trio edit a frozen list) --
    what the rule would have said about them is recorded, not applied. No select cache and no eval population is read."""
    m3b_compile = M3B_RUN.load_script("m3b_compile")
    if sorted(datasets) != sorted(PILOT):
        raise SystemExit("the screen runs on the three pilot fit carves together (training.pilot_datasets)")
    if dated_blocks(cfg, "contract_frozen"):
        raise SystemExit("contract_frozen_<date> is filed; the screen is not repeated (a different list is a new dated block)")
    core78, _, sha78 = m3b_core(cfg_m3b)
    pin_key, _ = pinned_raw_contract(cfg, core78)
    names = screen_input_columns(core78)
    raw_sha = sha_of_names(names)
    rule_text = str(cfg["information_contract_v2"]["screen"]).strip()
    rule_sha = hashlib.sha256(rule_text.encode("utf-8")).hexdigest()
    take = np.asarray([IDX[n] for n in names], dtype=np.int64)
    n_in = len(names)
    dirs = {name: screen_cache_dir(name) for name in datasets}
    total_rows = sum(int(np.load(d / "pool_ptr.npy")[-1]) for d in dirs.values())
    stride = max(1, math.ceil(total_rows / m3b_compile.SCREEN_SAMPLE_ROWS))
    per_dataset, samples = {}, []
    for name, d in dirs.items():
        stats = column_stats_v2(d, take)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        scalars = np.load(d / "scalars.npy", mmap_mode="r")
        samples.append(np.asarray(scalars[::stride], dtype=np.float64)[:, take])
        avail_of = dict(zip(names, stats["availability"].round(6).tolist()))
        per_dataset[name] = {"fit_rows": stats["rows"], "availability": stats["availability"].round(6).tolist(),
                             "variance": stats["variance"].round(8).tolist(),
                             "block_availability": {b: {m: avail_of[m] for m in MASK_COLUMNS if BLOCK_OF[m] == b} for b in BLOCKS},
                             "constant_columns": [c for c, v in zip(names, stats["variance"]) if v <= 0],
                             "cost": {"ms_per_query": meta.get("ms_per_query"), "group_seconds_per_1000_queries": meta.get("group_seconds_per_1000_queries"),
                                      "candidates_mean": meta.get("candidates_mean"), "bytes_per_row": 2 * N_COLUMNS}}
        log(f"   {name}: {stats['rows']} fit rows screened ({n_in} columns)")
    sample = np.concatenate(samples)
    C, pairs = m3b_compile.spearman_duplicates(sample, m3b_compile.DUPLICATE_RHO)
    avail = np.asarray([per_dataset[n]["availability"] for n in datasets])
    var = np.asarray([per_dataset[n]["variance"] for n in datasets])
    unavailable_everywhere = (avail < m3b_compile.UNAVAILABLE_BELOW).all(axis=0)
    zero_variance_everywhere = (var <= 0).all(axis=0)
    protected = np.arange(n_in) < M3B_CORE_SIZE
    later_duplicate = np.zeros(n_in, dtype=bool)
    for i, j, _ in pairs:
        if protected[i] or not (unavailable_everywhere[i] or zero_variance_everywhere[i] or later_duplicate[i]):
            later_duplicate[j] = True             # the later member of a pair whose earlier member survives
    would_drop = unavailable_everywhere | zero_variance_everywhere | later_duplicate
    dropped = would_drop & ~protected
    surviving = [c for c, d in zip(names, dropped) if not d]

    def reason(k: int) -> str:
        return ("unavailable_everywhere" if unavailable_everywhere[k] else "zero_variance_everywhere" if zero_variance_everywhere[k]
                else "later_member_of_duplicate_pair")

    reasons = {c: reason(k) for k, c in enumerate(names) if dropped[k]}
    core_would = {c: reason(k) for k, c in enumerate(names) if would_drop[k] and protected[k]}
    core_sha = sha_of_names(surviving)
    surviving_v2 = [c for c in surviving if c in BLOCK_OF]
    screen = {"utc": utc(), "sample_rows": int(sample.shape[0]), "stride": stride, "total_fit_rows": total_rows,
              "duplicate_threshold": m3b_compile.DUPLICATE_RHO, "unavailable_below": m3b_compile.UNAVAILABLE_BELOW,
              "rule": rule_text, "screen_rule_sha256": rule_sha, "raw_contract_pinned_in": pin_key, "raw_contract_sha256": raw_sha,
              "columns": names, "m3b_core": {"size": M3B_CORE_SIZE, "sha256": sha78, "protected": True,
                                             "would_have_dropped_under_the_trio_statistics": core_would},
              "v2_columns": {"screened": N_V2, "depth_basis": N_DEPTH, "ordered_relation_path": N_ORDERED, "surviving": len(surviving_v2),
                             "depth_basis_surviving": sum(c not in ORDERED_COLUMNS for c in surviving_v2),
                             "ordered_relation_path_surviving": sum(c in ORDERED_COLUMNS for c in surviving_v2)},
              "constant_columns_everywhere": [c for c, z in zip(names, zero_variance_everywhere) if z],
              "unavailable_everywhere": [c for c, z in zip(names, unavailable_everywhere) if z],
              "per_dataset": per_dataset,
              "duplicate_pairs": [{"earlier": names[i], "later": names[j], "abs_spearman": round(r, 4)} for i, j, r in pairs],
              "dropped": reasons, "surviving": surviving, "core_contract_sha256": core_sha,
              "abs_spearman_max_offdiag": {names[i]: round(float(np.max(np.delete(C[i], i))), 4) for i in range(n_in)},
              "read": "outputs/universal_v2/cache/<dataset>/fit only (check_3)"}
    (OUT / "feature_screen.json").write_text(json.dumps(screen, indent=1), encoding="utf-8")
    block = {"name": "UNIVERSAL_V2_CORE_CONTRACT", "utc": screen["utc"], "raw_contract_pinned_in": pin_key, "raw_contract_sha256": raw_sha,
             "screened_columns": n_in, "surviving_columns": len(surviving),
             "m3b_core_first": {"size": M3B_CORE_SIZE, "sha256": sha78, "always_survive": True,
                                "would_have_dropped_under_the_trio_statistics": core_would},
             "v2_columns_screened": N_V2, "depth_basis_screened": N_DEPTH, "ordered_relation_path_screened": N_ORDERED,
             "v2_columns_surviving": len(surviving_v2), "depth_basis_surviving": screen["v2_columns"]["depth_basis_surviving"],
             "ordered_relation_path_surviving": screen["v2_columns"]["ordered_relation_path_surviving"], "dropped": reasons,
             "constant_columns_everywhere": screen["constant_columns_everywhere"], "unavailable_everywhere": screen["unavailable_everywhere"],
             "duplicate_pairs": screen["duplicate_pairs"], "sha256_of_comma_joined_surviving_names": core_sha, "surviving": surviving,
             "screen_file": "outputs/universal_v2/feature_screen.json", "rule_applied": rule_text, "screen_rule_sha256": rule_sha}
    text = yaml.safe_dump(block, sort_keys=False, width=110)
    (OUT / "universal_v2_core_contract_block.yaml").write_text(text, encoding="utf-8")
    log(f"screen: {n_in} columns in (raw contract {raw_sha[:12]}, pinned by {pin_key}), {len(surviving)} survive ({len(surviving_v2)} of the {N_V2} v2 "
        f"columns), sha {core_sha[:12]}; dropped {len(reasons)}; M3B core would-have-dropped {len(core_would)} (recorded, not applied)")
    return screen


# ── file: the dated blocks, each copied from its sidecar once ────────────────


def cache_hashes(cache_dir: Path) -> dict:
    """Every array file of one compiled carve hashed (meta.json excluded: it is hashed on its own), and one combined
    sha256 over the sorted name:sha lines (amendment 2 step 3 item 7: the three compiled caches are hashed)."""
    files = sorted(p for p in cache_dir.iterdir() if p.is_file() and p.name != "meta.json")
    per_file = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files}
    combined = hashlib.sha256(LF.join(f"{name}:{v['sha256']}" for name, v in per_file.items()).encode("utf-8")).hexdigest()
    return {"files": per_file, "combined_sha256": combined, "bytes": sum(v["bytes"] for v in per_file.values())}


def stage_file(cfg: dict, date: str, which: list[str], config: Path = CONFIG, log=print) -> None:
    """order_of_operations 3, 4 and 6: contract_frozen_<date>, timing_<date> and pilot_gate_record_<date>, each copied
    from its sidecar with the sidecar's sha256; run_record_<date> assembles the fit, eval and gate records;
    replication_record_<date> (amendment 4) assembles the replication fits, the supplement evals and the two statuses
    and moves no status line."""
    if not re.match("^[0-9]{4}_[0-9]{2}_[0-9]{2}$", date):
        raise SystemExit(f"--date {date}: YYYY_MM_DD")
    unknown = [w for w in which if w not in ("contract", "timing", "gate", "run", "replication")]
    if unknown:
        raise SystemExit(f"--which {unknown}: contract, timing, gate, run or replication")
    filed = utc()
    if "contract" in which:
        path = OUT / "universal_v2_core_contract_block.yaml"
        if not path.exists():
            raise SystemExit("no screen output to file; run --stage screen first")
        block = yaml.safe_load(path.read_text(encoding="utf-8"))
        compiled: dict = {}
        for name in PILOT:
            compiled[name] = {}
            for kind in ("fit", "select"):
                mp = CACHE / name / kind / "meta.json"
                if not mp.exists():
                    raise SystemExit(f"{name}/{kind}: no compiled cache; compile every carve before filing")
                m = json.loads(mp.read_text(encoding="utf-8"))
                if int(m.get("n_columns", -1)) != N_COLUMNS or m.get("feature_contract") != CONTRACT_NAME:
                    raise SystemExit(f"{name}/{kind}: a cache of {m.get('n_columns')} columns / {m.get('feature_contract')}; not this contract")
                hashes = cache_hashes(mp.parent)
                compiled[name][kind] = {"queries": m["n_queries"], "rows": m["n_rows"], "zero_gold_excluded": m["population"]["zero_gold_excluded"],
                                        "kept_ids_sha256": m["population"]["ids_sha256"], "carve_sha256_declared": m["carve_sha256_declared"],
                                        "candidates_mean": round(float(m["candidates_mean"]), 1), "queries_with_no_gold_in_pool": m["queries_with_no_gold_in_pool"],
                                        "compile_seconds": m.get("compile_seconds"), "ms_per_query": m["ms_per_query"], "queries_per_second": m.get("queries_per_second"),
                                        "group_seconds_per_1000_queries": m.get("group_seconds_per_1000_queries"), "peak_rss_bytes": m.get("peak_rss_bytes"),
                                        "seeds_added_mean": m.get("seeds_added_mean"), "relation_slots": m.get("relation_slots"), "relation_table": m.get("relation_table"),
                                        "diagnostics": m.get("diagnostics"), "cache_bytes": m["bytes"], "cache_files_sha256": hashes["files"],
                                        "cache_combined_sha256": hashes["combined_sha256"], "meta_sha256": sha256_file(mp)}
        caches_sha = hashlib.sha256(LF.join(f"{n}/{k}:{compiled[n][k]['cache_combined_sha256']}" for n in PILOT for k in ("fit", "select")).encode("utf-8")).hexdigest()
        block = {"status": "FROZEN", "filed_utc": filed, **block, "screen_file_sha256": sha256_file(OUT / "feature_screen.json"),
                 "feature_contract_file": "outputs/universal_v2/feature_contract.json",
                 "feature_contract_sha256": sha256_file(OUT / "feature_contract.json"),
                 "hashes": {"raw_contract_sha256": block["raw_contract_sha256"], "screen_rule_sha256": block["screen_rule_sha256"],
                            "surviving_columns_sha256": block["sha256_of_comma_joined_surviving_names"], "six_caches_combined_sha256": caches_sha},
                 "compile_record": compiled,
                 "after_this_block": "the arms read this list and nothing else; the eval populations may now be compiled per query (check_3); "
                                     "a different list is a new dated block with its reason"}
        append_block(cfg, f"contract_frozen_{date}", block, f"UNIVERSAL_V2_CORE_CONTRACT, filed {filed} from outputs/universal_v2/feature_screen.json", config)
        log(f"filed contract_frozen_{date}")
    if "timing" in which:
        path = OUT / "timing.json"
        if not path.exists():
            raise SystemExit("no timing.json; run --stage timing first")
        rec = json.loads(path.read_text(encoding="utf-8"))
        block = {"status": "MEASURED_BEFORE_ANY_FIT", "filed_utc": filed, **rec, "output": "outputs/universal_v2/timing.json", "output_sha256": sha256_file(path)}
        append_block(cfg, f"timing_{date}", block, f"timing run (compute.timing_run), filed {filed} from outputs/universal_v2/timing.json", config)
        log(f"filed timing_{date}")
    if "gate" in which:
        path = OUT / "gate_record.json"
        if not path.exists():
            raise SystemExit("no gate_record.json; run --stage gate first")
        rec = json.loads(path.read_text(encoding="utf-8"))
        block = {"status": "READ_ONCE_ON_V2_GATE", "filed_utc": filed, **rec, "output": "outputs/universal_v2/gate_record.json", "output_sha256": sha256_file(path)}
        append_block(cfg, f"pilot_gate_record_{date}", block, f"pilot gate record (pilot_gate, V2_GATE only), filed {filed} from outputs/universal_v2/gate_record.json", config)
        moved = set_status(config, "PILOT_GATE_READ", ("DECLARED_NOT_RUN",))
        log(f"filed pilot_gate_record_{date}; status {moved} -> PILOT_GATE_READ")
    if "run" in which:
        fits = {}
        for p in sorted(FITS.glob("*.json")) if FITS.exists() else []:
            r = json.loads(p.read_text(encoding="utf-8"))
            fits[r["key"]] = {k: r.get(k) for k in ("arm", "seed", "parameters", "best_epoch", "epochs_run", "best_select_macro_recall5", "seconds",
                                                    "steps", "threads", "peak_rss_bytes", "state_sha256", "utc")}
            fits[r["key"]]["record_sha256"] = sha256_file(p)
        gate = read_json(OUT / "gate_record.json")
        if gate is None or not (OUT / "held_record.json").exists():
            raise SystemExit("the run record follows the gate record and the held record (order_of_operations 7); one of them is missing")
        evals = {}
        for p in sorted(EVAL.glob("*.json")) if EVAL.exists() else []:   # the seed-0 records and every supplement record, unsharded
            if p.name.endswith("_query_ids.json") or "__shard" in p.name:
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            evals[p.stem] = {k: r.get(k) for k in ("dataset", "supplement", "scorers", "utc", "queries", "ids_sha256", "seconds", "ms_per_query",
                                                   "threads", "peak_rss_bytes", "halves")}
            evals[p.stem]["record_sha256"] = sha256_file(p)
            evals[p.stem]["arrays_sha256"] = sha256_file(EVAL / f"{p.stem}.npz")
        doc = DOC
        if not doc.exists():
            raise SystemExit("docs/UNIVERSAL_V2_PILOT.md is written before the run record (scripts/universal_v2_report.py --stage doc)")
        family = gate.get("family_outcome") or family_outcome(gate["selection"], gate["outcome"])
        report_record = read_json(OUT / "report_run_record.json")
        if report_record is None:
            raise SystemExit("no report_run_record.json: the document and its seed confirmation precede the run record")
        terminal = terminal_state(family, report_record.get("seed_confirmation") or {})
        overall = terminal["terminal"]
        block = {"filed_utc": filed, "outcome": gate["outcome"], "family_outcome": family, "terminal_state": terminal, "fits": fits, "evals": evals,
                 "selection_sha256": sha256_file(OUT / "selection.json") if (OUT / "selection.json").exists() else None,
                 "gate_record_sha256": sha256_file(OUT / "gate_record.json") if (OUT / "gate_record.json").exists() else None,
                 "held_record_sha256": sha256_file(OUT / "held_record.json") if (OUT / "held_record.json").exists() else None,
                 "doc": "docs/UNIVERSAL_V2_PILOT.md", "doc_sha256": lf_sha256(doc) if doc.exists() else None,
                 "incidents": [], "note": "incidents and machine-health lines are appended by the review, dated; nothing above is edited"}
        append_block(cfg, f"run_record_{date}", block, f"run record, filed {filed} from the sidecars", config)
        moved = set_status(config, f"RUN_{overall}", ("PILOT_GATE_READ",))
        log(f"filed run_record_{date}; status {moved} -> RUN_{overall} (GNN_GATE {family['GNN_GATE']} {terminal['family_final']['GNN_GATE']}, "
            f"TWIN_GATE {family['TWIN_GATE']} {terminal['family_final']['TWIN_GATE']})")
    if "replication" in which:
        rep_key, rep = replication_amendment(cfg)
        rec = read_json(OUT / "replication_record.json")
        if rec is None:
            raise SystemExit("no replication_record.json: scripts/universal_v2_report.py --stage replication precedes the record (amendment 4 execution)")
        held = read_json(OUT / "replication_held_record.json")
        if rec["passing_families"] and held is None:
            raise SystemExit("a REPLICATION_PASS family reads the held half once (replication_held_record.json) before the record is filed")
        if not rec["passing_families"] and held is not None:
            raise SystemExit("replication_held_record.json exists although no family is REPLICATION_PASS; refusing")
        report = read_json(OUT / "replication_report_record.json")
        if not REPLICATION_DOC.exists() or report is None:
            raise SystemExit("docs/UNIVERSAL_V2_REPLICATION.md and replication_report_record.json precede the record (--stage replication_doc)")
        if report["doc_sha256_lf"] != lf_sha256(REPLICATION_DOC) or report["replication_record_sha256"] != sha256_file(OUT / "replication_record.json"):
            raise SystemExit("the document or the replication record changed after the report record; render again first")
        fits, spent, replication_hours = {}, 0.0, 0.0
        for p in sorted(FITS.glob("*.json")) if FITS.exists() else []:
            spent += float(json.loads(p.read_text(encoding="utf-8")).get("seconds", 0.0)) / 3600
        for fam, arm in rec["selected"].items():
            for seed in (0, 1, 2):
                k = fit_key(arm, seed)
                r = json.loads((FITS / f"{k}.json").read_text(encoding="utf-8"))
                fits[k] = {kk: r.get(kk) for kk in ("arm", "seed", "parameters", "best_epoch", "epochs_run", "best_select_macro_recall5", "seconds",
                                                    "steps", "threads", "peak_rss_bytes", "state_sha256", "utc")}
                fits[k]["record_sha256"] = sha256_file(FITS / f"{k}.json")
                fits[k]["authorised_by"] = (r.get("replication") or {}).get("block")
                if seed != 0:
                    replication_hours += float(r["seconds"]) / 3600
        evals, lane_hours = {}, 0.0
        for p in sorted(EVAL.glob("*__more_*.json")) if EVAL.exists() else []:
            if p.name.endswith("_query_ids.json") or "__shard" in p.name:
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            evals[p.stem] = {k: r.get(k) for k in ("dataset", "supplement", "scorers", "utc", "queries", "ids_sha256", "seconds", "ms_per_query",
                                                   "threads", "peak_rss_bytes", "halves", "merged_from")}
            evals[p.stem]["record_sha256"] = sha256_file(p)
            evals[p.stem]["arrays_sha256"] = sha256_file(EVAL / f"{p.stem}.npz")
            lane_hours += float(r["seconds"]) / 3600
        try:
            out = subprocess.run(["git", "log", "--format=%h %s", f"{rep['original_pilot_commit']}..HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
            commits = [line for line in out.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.CalledProcessError) as e:   # the sandbox names no real commit
            commits = [f"not read: {e}"]
        incidents = read_json(OUT / "replication_incidents.json") or []
        block = {"filed_utc": filed, "amendment": rep_key, "original_pilot_status": rec["original_pilot_status"], "original_pilot_commit": rep["original_pilot_commit"],
                 "replication_status": rec["replication_status"], "family_status": rec["family_status"], "passing_families": rec["passing_families"],
                 "held_half_read_for": rec["passing_families"], "never": rec.get("never", "PILOT_PASS"), "selected": rec["selected"], "fits": fits, "evals": evals,
                 "compute": {"replication_fit_hours": round(replication_hours, 3), "fit_hours_spent_in_total": round(spent, 3), "ceiling_fit_hours": FIT_HOURS_CEILING,
                             "supplement_eval_lane_hours": round(lane_hours, 3)},
                 "replication_record_sha256": sha256_file(OUT / "replication_record.json"),
                 "replication_held_record_sha256": sha256_file(OUT / "replication_held_record.json") if held is not None else None,
                 "doc": REPLICATION_DOC.relative_to(ROOT).as_posix() if REPLICATION_DOC.is_relative_to(ROOT) else REPLICATION_DOC.as_posix(),
                 "doc_sha256": lf_sha256(REPLICATION_DOC), "report_record_sha256": sha256_file(OUT / "replication_report_record.json"),
                 "commits_since_the_pilot_closed": commits, "status_line": status_line(config), "status_line_moved": False, "incidents": incidents,
                 "note": "the original status stays as filed (amendment 4 status_vocabulary.status_line); incidents are copied from "
                         "outputs/universal_v2/replication_incidents.json as logged during the run; nothing above is edited"}
        append_block(cfg, f"replication_record_{date}", block, f"post-pilot replication record (amendment 4), filed {filed} from the sidecars; the status line is not moved", config)
        log(f"filed replication_record_{date}: ORIGINAL PILOT STATUS {block['original_pilot_status']}; POST-PILOT REPLICATION STATUS {block['replication_status']} "
            f"({block['family_status']}); status line {block['status_line']} not moved")


# ── fits ─────────────────────────────────────────────────────────────────────


def fit_key(arm: str, seed: int) -> str:
    return f"{arm}__H{HIDDEN}__s{seed}"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def make_model(arm: str, inputs: dict, bank, selected_gnn: str | None = None) -> torch.nn.Module:
    """build_arm at the declared width; the served relation bank set on the GNN arms; the budget checked."""
    model = build_arm(arm, inputs, hidden=HIDDEN, dropout=DROPOUT, heads=HEADS, steps=STEPS, d_r=D_R, d_p=D_P, selected_gnn=selected_gnn)
    if hasattr(model, "set_relation_bank"):
        model.set_relation_bank(bank)
    check_parameter_budget(arm, model)
    return model


def stage_timing(cfg: dict, inputs: dict, carves: dict, bank, training: dict, log=print) -> dict:
    """compute.timing_run: seconds per training batch for every arm on three mixed batches of the declared
    sampler, then one full epoch of u_gnn_v2_ef (the most expensive arm) through the pinned fit_model with
    max_epochs 1. The weights are discarded and no select number is kept; timing_<date> is filed before the
    seed-0 matrix; the fallback fires if the epoch exceeds three hours."""
    if (OUT / "timing.json").exists():
        log("timing.json exists, not repeated")
        return read_json(OUT / "timing.json")
    if FITS.exists() and any(FITS.glob("*.json")):
        raise SystemExit("a fit record exists; the timing run precedes the seed-0 matrix (order_of_operations 4)")
    batch_size = training["batch_size"]
    arms = {}
    for arm in ARMS:
        torch.manual_seed(0)
        stand_in = "u_gnn_v2_ef" if arm == "u_gnn_v2_core78" else None
        model = make_model(arm, inputs, bank, selected_gnn=stand_in)
        fits = carves_for_arm(arm, carves)["fit"]
        names = sorted(fits)
        optimiser = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
        rng = np.random.default_rng(0)
        cursors = {n: [rng.permutation(fits[n].trainable), 0] for n in names}
        mixed = []
        for _ in range(3):
            t = time.perf_counter()
            batch = pack_parts(fits, draw_indices(fits, names, cursors, rng, batch_size, training["dataset_draw"]), FAMILIES)
            t_pack = time.perf_counter() - t
            model.train()
            t = time.perf_counter()
            loss = listwise_loss(model(batch), batch)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            mixed.append({"pack_s": round(t_pack, 3), "step_s": round(time.perf_counter() - t, 3), "nodes": int(batch.x.shape[0]),
                          "edges": int(batch.edge_attr.shape[0])})
        mean_s = float(np.mean([m["pack_s"] + m["step_s"] for m in mixed]))
        arms[arm] = {"parameters": parameter_count(model), "architecture": stand_in or arm, "mixed_batches": mixed,
                     "mean_seconds_per_batch": round(mean_s, 3), "projected_epoch_minutes": round(mean_s * training["batches_per_epoch"] / 60, 1)}
        log(f"   {arm}: {arms[arm]['parameters']} parameters, {mean_s:.2f} s/batch, {arms[arm]['projected_epoch_minutes']} min/epoch projected")
        del model, optimiser
    torch.manual_seed(0)
    model = make_model("u_gnn_v2_ef", inputs, bank)
    data = carves_for_arm("u_gnn_v2_ef", carves)

    def quiet(msg: str) -> None:   # the epoch line carries a select number; the timing run keeps the seconds only
        log("      epoch 0 finished (select number not kept in the timing run)" if "select macro" in msg else msg)

    t = time.time()
    _, record = fit_model(model, data["fit"], data["select"], seed=0, arm="u_gnn_v2_ef", config={"H": HIDDEN, "purpose": "timing"}, max_epochs=1,
                          batches_per_epoch=training["batches_per_epoch"], batch_size=batch_size, patience=training["patience"], lr=training["lr"],
                          weight_decay=training["weight_decay"], clip=training["clip"], dataset_draw=training["dataset_draw"], epoch_limit_s=None,
                          pack_workers=PACK["workers"], prefetch_depth=PACK["depth"], checkpoint=None, log=quiet)
    epoch = record.history[0]
    del model
    out = {"utc": utc(), "threads": torch.get_num_threads(), "batch_size": batch_size, "batches_per_epoch": training["batches_per_epoch"],
           "dataset_draw": training["dataset_draw"], "pack_workers": PACK["workers"], "prefetch_depth": PACK["depth"],
           "contract_block": inputs["contract_block"], "core_sha256": inputs["core_sha256"], "arms": arms,
           "full_epoch_u_gnn_v2_ef": {"epoch_seconds": epoch["seconds"], "steps": record.steps, "batches_skipped_no_gold": record.batches_skipped_no_gold,
                                      "wall_seconds": round(time.time() - t, 1), "select_evaluation_included": True, "weights": "discarded"},
           "peak_rss_gb": round(M3B_RUN.peak_rss_bytes() / 2**30, 2),
           "fallback": {"rule": str(cfg["compute"]["timing_run"]).strip(), "fires": bool(float(epoch["seconds"]) > 3 * 3600)}}
    (OUT / "timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"timing: u_gnn_v2_ef epoch {epoch['seconds']}s at {out['threads']} threads; fallback fires: {out['fallback']['fires']}")
    return out


def run_fit(arm: str, seed: int, inputs: dict, carves: dict, bank, training: dict, log=print, cfg: dict | None = None) -> dict:
    """One fit under training.rule, resumable through its checkpoint; the record and the weights are the durable
    objects. Refusals: an unknown arm or seed; u_gnn_v2_core78 before selection.json; seeds 1-2 of an arm
    without a gate pass (pilot_gate.on_pass) unless the dated post-pilot replication amendment names exactly that
    (arm, seed) at the status it applies at (amendment 4, replication_authorisation); a parameter count over budget;
    under the replication, a training rule or an architecture that differs from the seed-0 record of the arm."""
    if arm not in ARMS or seed not in (0, 1, 2):
        raise SystemExit(f"{arm} seed {seed}: not an arm of the declaration or not a declared seed")
    key = fit_key(arm, seed)
    FITS.mkdir(parents=True, exist_ok=True)
    rec_path = FITS / f"{key}.json"
    if rec_path.exists():
        log(f"   {key}: record exists, not repeated")
        return read_json(rec_path)
    selection = read_json(OUT / "selection.json")
    if arm == "u_gnn_v2_core78" and selection is None:
        raise SystemExit("u_gnn_v2_core78 takes the selected GNN architecture; selection.json is written first (order_of_operations 6)")
    replication = None
    if seed != 0:
        gate = read_json(OUT / "gate_record.json")
        if gate is None or not gate["verdict"].get(arm, {}).get("pass"):
            replication = None if cfg is None else replication_authorisation(cfg, arm, seed)
            if replication is None:
                raise SystemExit(f"seed {seed} of {arm}: seeds 1-2 are fitted only for an arm that passed its gate (pilot_gate.on_pass) "
                                 "or under a dated post-pilot replication amendment naming this arm and seed (amendment 4)")
    selected_gnn = None if selection is None else selection["gnn"]["arm"]
    guard = fit_hours_guard(arm, training, log=log)
    torch.manual_seed(seed)
    model = make_model(arm, inputs, bank, selected_gnn=selected_gnn)
    params = parameter_count(model)
    on78 = arm in ("u_gnn_v2_core78", "gat_universal_v1_trio")
    with_evidence = arm == "u_gnn_v2_ef" or (arm == "u_gnn_v2_core78" and selected_gnn == "u_gnn_v2_ef")
    frozen = {"parameters": params, "hidden": HIDDEN, "contract_block": inputs["contract_block"], "columns": M3B_CORE_SIZE if on78 else inputs["n_scalars"],
              "core_sha256": inputs["core78_sha256"] if on78 else inputs["core_sha256"],
              "evidence": (inputs["core78_evidence"] if on78 else inputs["evidence"]) if with_evidence else None,
              "evidence_substitutions": (inputs["core78_evidence_substitutions"] if on78 else inputs["evidence_substitutions"]) if with_evidence else None,
              "relation_bank": {"rows": bank.n_rows, "sha256": bank.sha256, "k_rel": K_REL}, "training": training}
    if replication is not None:   # amendment 4: the frozen protocol and the frozen architecture are the seed-0 record of the arm
        seed0 = replication.pop("seed0_record")
        differs = {k: {"this_fit": v, "seed_0": seed0.get(k)} for k, v in frozen.items() if seed0.get(k) != v}
        if differs:
            raise SystemExit(f"{key}: {sorted(differs)} differ from the seed-0 record of {arm}; the replication fits the frozen protocol and "
                             f"architecture only (amendment 4 hard stop): {differs}")
        replication["checks_against_seed_0"] = {k: "equal" for k in frozen}
        log(f"   {key}: replication under {replication["block"]}; protocol and architecture equal the seed-0 record")
    log(f"== fit {key}: {params} parameters")
    data = carves_for_arm(arm, carves)
    model, record = fit_model(model, data["fit"], data["select"], seed=seed, arm=arm, config={"H": HIDDEN}, max_epochs=training["max_epochs"],
                              batches_per_epoch=training["batches_per_epoch"], batch_size=training["batch_size"], patience=training["patience"],
                              lr=training["lr"], weight_decay=training["weight_decay"], clip=training["clip"], dataset_draw=training["dataset_draw"],
                              epoch_limit_s=training["epoch_limit_s"], pack_workers=PACK["workers"], prefetch_depth=PACK["depth"],
                              checkpoint=FITS / f"{key}.ckpt", log=log)
    torch.save(model.state_dict(), FITS / f"{key}.pt")
    out = {**asdict(record), "key": key, "hidden": HIDDEN, "parameters": params, "parameter_budget": PARAMETER_BUDGET[FAMILY_OF_ARM[arm]],
           "contract_block": frozen["contract_block"], "columns": frozen["columns"], "core_sha256": frozen["core_sha256"], "base": inputs["base"],
           "evidence": frozen["evidence"], "evidence_substitutions": frozen["evidence_substitutions"],
           "selected_gnn_architecture": selected_gnn if arm == "u_gnn_v2_core78" else None,
           "relation_bank": frozen["relation_bank"], "training": training, "utc": utc(),
           "threads": torch.get_num_threads(), "pack_workers": PACK["workers"], "prefetch_depth": PACK["depth"],
           "peak_rss_bytes": M3B_RUN.peak_rss_bytes(), "state_sha256": sha256_file(FITS / f"{key}.pt"), "ceiling_guard": guard,
           "replication": replication}
    rec_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (FITS / f"{key}.ckpt").unlink(missing_ok=True)   # the record and the weights are the durable objects
    log(f"   {key}: best epoch {record.best_epoch} select macro R@5 {record.best_select_macro_recall5:.4f} in {record.seconds:.0f}s")
    return out


def stage_select(cfg: dict, inputs: dict, log=print) -> dict:
    """arms.selection_behind_the_firewall: within each family the argmax of the seed-0 best select macro recall@5;
    ties to the smaller parameter count; written once, before any eval record exists; no eval number and no
    number of the other family enters it."""
    path = OUT / "selection.json"
    if path.exists():
        raise SystemExit("selection.json exists; it is written once")
    if EVAL.exists() and any(EVAL.iterdir()):
        raise SystemExit("an eval record exists; the selection is written before any eval population is scored")
    records = {}
    for arm in GNN_CANDIDATES + TWIN_CANDIDATES:
        rec = read_json(FITS / f"{fit_key(arm, 0)}.json")
        if rec is None:
            raise SystemExit(f"{arm}: no seed-0 fit record; the selection needs all four candidates")
        records[arm] = rec

    def family(candidates) -> dict:
        best = max(candidates, key=lambda a: (records[a]["best_select_macro_recall5"], -records[a]["parameters"]))
        return {"arm": best, "select_macro_recall5": records[best]["best_select_macro_recall5"], "parameters": records[best]["parameters"],
                "candidates": {a: {"select_macro_recall5": records[a]["best_select_macro_recall5"], "parameters": records[a]["parameters"],
                                   "best_epoch": records[a]["best_epoch"], "seconds": records[a]["seconds"]} for a in candidates}}

    selection = {"utc": utc(), "rule": cfg["arms"]["selection_behind_the_firewall"], "gnn": family(GNN_CANDIDATES), "twin": family(TWIN_CANDIDATES),
                 "contract_block": inputs["contract_block"], "core_sha256": inputs["core_sha256"], "eval_populations_scored_before_this_file": False}
    path.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    log(f"selection: GNN {selection['gnn']['arm']}, twin {selection['twin']['arm']}")
    return selection


# ── the eval pass ────────────────────────────────────────────────────────────


def load_models(inputs: dict, selection: dict, bank) -> dict[str, torch.nn.Module]:
    """Every fitted arm under outputs/universal_v2/fits/ (record + weights), keyed by its fit key."""
    models: dict[str, torch.nn.Module] = {}
    for p in sorted(FITS.glob("*.json")) if FITS.exists() else []:
        rec = read_json(p)
        pt = FITS / f"{rec['key']}.pt"
        if not pt.exists():
            continue
        if rec["arm"] == "u_gnn_v2_core78" and rec.get("selected_gnn_architecture") != selection["gnn"]["arm"]:
            raise SystemExit(f"{rec['key']}: fitted on {rec.get('selected_gnn_architecture')}, the selection names {selection['gnn']['arm']}")
        m = make_model(rec["arm"], inputs, bank, selected_gnn=selection["gnn"]["arm"])
        m.load_state_dict(torch.load(pt, map_location="cpu"))
        m.eval()
        models[rec["key"]] = m
    if not models:
        raise SystemExit("no fitted models under outputs/universal_v2/fits/")
    return models


def arm_view(model, batch, inputs: dict):
    """The packed v2 batch as each arm reads it: the control sees the M3B 78 and the M3B eight edge columns
    (the pinned pack, as a test holds); the ablation sees the M3B 78 with the v2 edges; the v2 arms the batch."""
    arm = getattr(model, "arm", "")
    if arm == "gat_universal_v1_trio":
        return batch_view(batch, inputs["core78_positions"], 8)
    if arm == "u_gnn_v2_core78":
        return batch_view(batch, inputs["core78_positions"], N_EDGE_FEATURES_V2)
    return batch


def _segment_mean(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    counts = np.diff(ptr)
    return np.add.reduceat(values, ptr[:-1]) / np.maximum(counts, 1)


@torch.no_grad()
def mechanism(model, batch, scores: torch.Tensor, ptr: np.ndarray) -> dict[str, np.ndarray]:
    """measurement.mechanism_readouts per query, after a forward pass: mean |delta_s| / mean |base_z| over the
    pool and whether the top-1 leaves the fixed base (delta_s_magnitude); the step gates and the evidence gates
    (u_gnn_v2, u_gnn_v2_ef, u_gnn_v2_core78); the block gates (u_mlp_v2_mix)."""
    if isinstance(model, UGNNv2):
        base_z, corr = model.last_base_z, model.last_correction
    else:
        _, base_z = model.input(batch)
        corr = scores - model.readout.base_weight * base_z
    base_np, corr_np, score_np = base_z.cpu().numpy().astype(np.float64), corr.cpu().numpy().astype(np.float64), scores.cpu().numpy()
    out = {"delta_ratio": _segment_mean(np.abs(corr_np), ptr) / (_segment_mean(np.abs(base_np), ptr) + 1e-12)}
    top_model = np.asarray([int(np.argmax(score_np[a:b])) for a, b in zip(ptr[:-1], ptr[1:])])
    top_base = np.asarray([int(np.argmax(base_np[a:b])) for a, b in zip(ptr[:-1], ptr[1:])])
    out["top1_changed"] = (top_model != top_base).astype(np.float64)
    if isinstance(model, UGNNv2):
        for t in range(model.last_gates.shape[0]):
            out[f"gate_step{t + 1}"] = _segment_mean(model.last_gates[t].cpu().numpy().astype(np.float64), ptr)
        if model.last_gates2 is not None:
            for t in range(model.last_gates2.shape[0]):
                out[f"gate2_step{t + 1}"] = _segment_mean(model.last_gates2[t].cpu().numpy().astype(np.float64), ptr)
    gates = getattr(model, "last_block_gates", None)
    if gates is not None:
        g = gates.cpu().numpy().astype(np.float64)
        for b in range(g.shape[1]):
            out[f"block_gate{b}"] = g[:, b]
    return out


def half_labels(name: str, ds, split: str, ids: list[str]) -> np.ndarray:
    """check_1: True = V2_GATE, False = V2_HELD_CONFIRMATION per query, from the served rows through in_v2_gate."""
    by = {r["query_id"]: r for r in ds.queries(split)}
    return np.asarray([in_v2_gate(name, by[q]) for q in ids], dtype=bool)


def declared_half_counts(cfg: dict, name: str) -> dict | None:
    """The filed counts of the halves (the latest amendment carrying check_1_the_halves.split_rule_amended.counts)."""
    for key in reversed(dated_blocks(cfg, "amendment")):
        block = cfg[key] if isinstance(cfg[key], dict) else {}
        counts = block.get("check_1_the_halves", {}).get("split_rule_amended", {}).get("counts", {})
        if name in counts:
            return counts[name]
    return None


def gold_distance_struct(scalars: np.ndarray, gold_local: np.ndarray) -> int:
    """The nearest in-pool gold's BFS bucket from the seeds in the STRUCT view (0..3, 4 = unreached; -1 = no gold in
    the pool), from the M3B topology one-hots (measurement.slices_reported.gold_distance)."""
    if gold_local.size == 0:
        return -1
    cols = [IDX[f"dist{d}_STRUCT"] for d in range(4)] + [IDX["distunreached_STRUCT"]]
    return int(np.argmax(scalars[gold_local][:, cols], axis=1).min())


def m3b_fixed_rrf_agreement(name: str, rrf: dict, shard, full_n: int) -> dict | None:
    """The frozen M3B eval arrays of the same population: the v2 pass's fixed rrf must reproduce M3B's fixed:rrf
    per query (same pools, same column, same order) -- every reference number is then on the same queries."""
    path = M3B_OUT / "eval" / f"{name}.npz"
    if not path.exists():
        return None
    diffs = {}
    with np.load(path) as z:
        for m in ("recall@5", "hit@1", "gold_in_pool", "gold_total", "pool_size"):
            ref = z[f"fixed:rrf/{m}"]
            if ref.size != full_n:
                raise SystemExit(f"{name}: the M3B arrays hold {ref.size} queries, this population {full_n}")
            if shard is not None:
                ref = ref[shard[0]::shard[1]]
            diffs[m] = float(np.max(np.abs(ref.astype(np.float64) - rrf[m]))) if ref.size else 0.0
    if max(diffs.values()) > 1e-9:
        raise SystemExit(f"{name}: the fixed rrf disagrees with the frozen M3B arrays {diffs}; the population or the pools differ; refusing")
    return {"file": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix(), "max_abs_diff": diffs, "ok": True}


def audit_scorers(arrays: dict, scorers: list[str], where: str) -> dict:
    """measurement.audit: the M3B MRR audit verbatim on every scorer before any record is written."""
    audits = {}
    for s in scorers:
        audit = mrr_audit(arrays[f"{s}/first_gold_rank"], arrays[f"{s}/mrr"])
        hit_le_mrr = bool((arrays[f"{s}/hit@1"] <= arrays[f"{s}/mrr"] + 1e-12).all()) and bool((arrays[f"{s}/mrr"] <= 1.0 + 1e-12).all())
        audit["hit1_le_mrr_le_1"] = hit_le_mrr
        audit["ok"] = audit["max_abs_diff"] <= 1e-12 and hit_le_mrr
        audits[s] = audit
        if not audit["ok"]:
            raise SystemExit(f"{where}/{s}: MRR audit failed: {audit}")
    return audits


def gate_summary(arrays: dict, scorers: list[str], half: np.ndarray) -> dict:
    """The record's summary is on V2_GATE only; the held half is summarised by the report stage, once."""
    return {s: {m: round(float(arrays[f"{s}/{m}"][half].mean()), 4) for m in ("recall@1", "recall@5", "recall@20", "hit@1", "mrr", "full_coverage@5")}
            for s in scorers}


@torch.no_grad()   # scoring only, as evaluate_carve: no autograd graph, outputs convert to numpy directly
def eval_dataset(name: str, cfg: dict, cfg_m3b: dict, cfg_h: dict, inputs: dict, models: dict, context, ds, pkg, m3b_compile, m3b_contract,
                 chunk_nodes: int, shard: tuple[int, int] | None = None, only_keys: list[str] | None = None, tag: str | None = None,
                 supplement: dict | None = None, log=print) -> dict:
    """scripts/m3b_run.eval_dataset under a v2 name: the M3B eval population (digest checked against
    m3b_incumbents), compiled per query under the v2 contract, scored by every fitted arm and the fixed
    scorers, with the half label per query (check_1), the slices and the mechanism readouts. The summary
    is on V2_GATE only. A supplement pass (only_keys, tag) scores later fits -- seeds 1-2 after the gate -- on the same
    population into <name>__<tag> beside the seed-0 record; the M3B fixed rrf agreement pins it to the same queries."""
    if only_keys is not None:
        models = {k: models[k] for k in only_keys}
    m3a, canonical, served, freeze = pkg
    key_m3b, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    split = cfg_m3b["populations"]["eval_splits"][name]
    declared = cfg["m3b_incumbents"]["eval_populations_reused_here"][name]
    t0 = time.time()
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]) or split != declared["split"]:
        raise SystemExit(f"{name}: not the M3B eval population (m3b_incumbents.eval_populations_reused_here); refusing")
    half_full = half_labels(name, ds, split, pop.ids)
    counts = declared_half_counts(cfg, name)
    if counts is not None and (int(half_full.sum()) != int(counts["V2_GATE"]) or int((~half_full).sum()) != int(counts["V2_HELD_CONFIRMATION"])):
        raise SystemExit(f"{name}: halves {int(half_full.sum())} / {int((~half_full).sum())} are not the filed counts {counts}; refusing")
    full_digest, full_n = pop.digest, int(pop.idx.size)
    half = half_full
    if shard is not None:   # queries k::N of the eval population, prepared and compiled by this process alone
        k, N = shard
        pop.ids, pop.idx, pop.golds, half = pop.ids[k::N], pop.idx[k::N], pop.golds[k::N], half_full[k::N]
    measure_latency = shard is None or shard[0] == 0
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    sizes = np.asarray([p.size for p in prep.pools])
    n = int(pop.idx.size)
    log(f"   {name}: {n} eval queries ({pop.zero_gold_excluded} zero-gold excluded), V2_GATE {int(half.sum())}, pools mean {sizes.mean():.0f}, "
        f"prepared in {time.time() - t0:.0f}s")
    ks = tuple(int(k) for k in cfg_h["retrieval_pools"]["ks"])
    ceiling, _ = m3a.cell(prep.pools, m3a.golds_ragged(pop.golds), int(ds.n_nodes), ks, dataset=name, population="eval", pool=frozen["per_dataset"][name]["pool"])
    columns = inputs["column_indices"]
    fixed = list(FIXED_SCORERS)
    scorers = list(models) + [f"fixed:{c}" for c in fixed]
    arrays = {f"{s}/{m}": np.zeros(n) for s in scorers for m in METRIC_NAMES}
    arrays["gold_dist_struct"] = np.full(n, -1, dtype=np.int64)
    arrays["hop"] = np.asarray([int(q.split(":")[1][0]) for q in pop.ids], dtype=np.int64) if name == "metaqa" else np.zeros(n, dtype=np.int64)
    latency = {"compile": [], "pack": [], **{k: [] for k in models}}
    chunk = max(1, int(chunk_nodes // max(sizes.mean(), 1)))
    t_loop = time.time()
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        qds, gold_locals = [], []
        for i in idx:
            t = time.perf_counter()
            E = context.nodes.read(prep.pools[i])
            inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
            compiled = compile_query_v2(inp, prep.pools[i], prep.seeds[i], context.stores, context.nodes, context.rel_table, embeddings=E)
            gold_local = m3b_compile.gold_local_of(prep.pools[i], pop.golds[i])
            if measure_latency and i < LATENCY_QUERIES:
                latency["compile"].append(time.perf_counter() - t)
            qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[i],
                        "seeds": compiled.seeds_local, "gold": gold_local, "gold_total": int(pop.golds[i].size), "emb": E})
            gold_locals.append(gold_local)
            arrays["gold_dist_struct"][i] = gold_distance_struct(compiled.scalars, gold_local)
            for c in fixed:
                r = rank_metrics(compiled.scalars[:, IDX[c]], gold_local, int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    arrays[f"fixed:{c}/{m}"][i] = r[m]
        if measure_latency and start < LATENCY_QUERIES:   # cold per-query latency, batch of one, on the first queries
            for j, i in enumerate(idx):
                if i >= LATENCY_QUERIES:
                    break
                t = time.perf_counter()
                single = pack_queries_v2([qds[j]], context)
                latency["pack"].append(time.perf_counter() - t)
                for key, model in models.items():
                    t = time.perf_counter()
                    model(arm_view(model, single, inputs))
                    latency[key].append(time.perf_counter() - t)
        batch = pack_queries_v2(qds, context)
        ptr = batch.qptr.numpy()
        for key, model in models.items():
            view = arm_view(model, batch, inputs)
            scores_t = model(view)
            scores = scores_t.cpu().numpy()
            for j, i in enumerate(idx):
                r = rank_metrics(scores[ptr[j]:ptr[j + 1]], gold_locals[j], int(pop.golds[i].size))
                for m in METRIC_NAMES:
                    arrays[f"{key}/{m}"][i] = r[m]
            for mname, values in mechanism(model, view, scores_t, ptr).items():
                arrays.setdefault(f"{key}/{mname}", np.zeros(n))[idx] = values
        if (start // chunk) % 20 == 0:
            done = min(start + chunk, n)
            log(f"      {name}: {done}/{n} queries, {(time.time() - t_loop) / done * 1000:.0f} ms/query")
    arrays["pool_size"] = sizes.astype(np.int64)
    arrays["half"] = half.astype(bool)
    return write_eval_record(name, arrays, scorers, pop, prep, ceiling, latency, shard, full_n, full_digest, key_m3b, frozen, construction,
                             chunk, t0, t_loop, freeze, inputs, log, file_base=name if tag is None else f"{name}__{tag}", supplement=supplement)


def write_eval_record(name, arrays, scorers, pop, prep, ceiling, latency, shard, full_n, full_digest, key_m3b, frozen, construction, chunk, t0, t_loop,
                      freeze, inputs, log=print, file_base: str | None = None, supplement: dict | None = None) -> dict:
    """The per-query arrays (both halves, labelled), the ceiling check against the headroom cell, the MRR audit,
    the agreement with the frozen M3B fixed rrf, and the record whose summary is on V2_GATE only."""
    EVAL.mkdir(parents=True, exist_ok=True)
    suffix = M3B_RUN.shard_suffix(shard)
    base = file_base or name
    n = int(pop.idx.size)
    sizes = arrays["pool_size"]
    first = scorers[0]
    from_arrays = M3B_RUN.ceiling_from_arrays(arrays[f"{first}/gold_in_pool"], arrays[f"{first}/gold_total"], sizes)
    if abs(from_arrays["recall_ceiling@5"] - float(ceiling["recall_ceiling@5"])) > 1e-9:
        raise SystemExit(f"{name}: ceiling from the per-query arrays {from_arrays['recall_ceiling@5']} != headroom cell {ceiling['recall_ceiling@5']}")
    audits = audit_scorers(arrays, scorers, name)
    agreement = m3b_fixed_rrf_agreement(name, {m: arrays[f"fixed:rrf/{m}"] for m in ("recall@5", "hit@1", "gold_in_pool", "gold_total", "pool_size")},
                                        shard, full_n)
    half = arrays["half"]
    np.savez_compressed(EVAL / f"{base}{suffix}.npz", **arrays)
    record = {"dataset": name, "utc": utc(), "queries": n, "zero_gold_excluded": pop.zero_gold_excluded, "ids_sha256": pop.digest,
              "supplement": supplement,
              "shard": None if shard is None else {"k": shard[0], "N": shard[1], "population_queries": int(full_n), "population_ids_sha256": full_digest},
              "contract_block": inputs["contract_block"], "core_sha256": inputs["core_sha256"], "m3b_contract_block": key_m3b,
              "pool": frozen["per_dataset"][name]["pool"], "construction": construction,
              "ceiling_as_compiled": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in ceiling.items()},
              "seeds_added_mean": float(prep.seeds_added.mean()), "scorers": scorers, "chunk_queries": chunk,
              "halves": {"V2_GATE": int(half.sum()), "V2_HELD_CONFIRMATION": int((~half).sum()), "label_array": "half (True = V2_GATE)",
                         "rule": "scripts/universal_v2_split_audit.py::in_v2_gate (check_1_the_halves.split_rule_amended)"},
              "seconds": round(time.time() - t0, 1), "ms_per_query": round(1000 * (time.time() - t_loop) / max(n, 1), 2),
              "latency": {k: M3B_RUN.percentiles(v) for k, v in latency.items()}, "peak_rss_bytes": M3B_RUN.peak_rss_bytes(),
              "threads": torch.get_num_threads(), "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "mrr_audit": audits,
              "m3b_fixed_rrf_agreement": agreement, "summary_half": "V2_GATE only; the held half is summarised by scripts/universal_v2_report.py once",
              "summary_V2_GATE": gate_summary(arrays, scorers, half)}
    (EVAL / f"{base}{suffix}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (EVAL / f"{base}{suffix}_query_ids.json").write_text(json.dumps(pop.ids), encoding="utf-8")
    log(f"   {base}: done in {record['seconds']}s; V2_GATE " + "; ".join(f"{s.split('__')[0]} R@5={v['recall@5']:.3f}" for s, v in record["summary_V2_GATE"].items()))
    return record


def merge_shards(name: str, log=print) -> dict | None:
    """scripts/m3b_run.merge_shards under a v2 name: the shard files interleaved back into population order,
    every array (metrics, labels, mechanism readouts) merged, the ceiling recomputed by the headroom's own
    functions, the summary on V2_GATE only."""
    from mp_retrieval.candidate_headroom import headroom_metrics
    from mp_retrieval.headroom_v2 import full_coverage_ceiling
    shards = sorted(p for p in EVAL.glob(f"{name}__shard*of*.json") if not p.name.endswith("_query_ids.json"))
    if not shards:
        return None
    records = [read_json(p) for p in shards]
    N = records[0]["shard"]["N"]
    have = sorted(r["shard"]["k"] for r in records)
    if have != list(range(N)):
        log(f"   {name}: shards {have} of {N} present, not merged")
        return None
    full_n, full_digest = records[0]["shard"]["population_queries"], records[0]["shard"]["population_ids_sha256"]
    for key in ("dataset", "supplement", "contract_block", "core_sha256", "m3b_contract_block", "pool", "construction", "scorers", "freeze_RECORD_SHA256"):
        if any(r[key] != records[0][key] for r in records):
            raise SystemExit(f"{name}: shard records disagree on {key}")
    if any(r["shard"]["population_queries"] != full_n or r["shard"]["population_ids_sha256"] != full_digest for r in records):
        raise SystemExit(f"{name}: shard records disagree on the population")
    by_k = {r["shard"]["k"]: r for r in records}
    ids: list = [None] * full_n
    merged: dict[str, np.ndarray] = {}
    for k in range(N):
        r = by_k[k]
        n_k = len(range(k, full_n, N))
        if r["queries"] != n_k:
            raise SystemExit(f"{name}: shard {k} holds {r['queries']} queries, slice k::N holds {n_k}")
        ids[k::N] = json.loads((EVAL / f"{name}__shard{k}of{N}_query_ids.json").read_text(encoding="utf-8"))
        with np.load(EVAL / f"{name}__shard{k}of{N}.npz") as z:
            for a in z.files:
                if a not in merged:
                    merged[a] = np.zeros(full_n, dtype=z[a].dtype)
                merged[a][k::N] = z[a]
    if any(i is None for i in ids) or m3b_pools.ids_digest(ids) != full_digest:
        raise SystemExit(f"{name}: merged query ids do not reproduce the population digest")
    scorers = records[0]["scorers"]
    sizes = merged["pool_size"]
    ks = tuple(sorted(int(k.split("@")[1]) for k in records[0]["ceiling_as_compiled"] if k.startswith("recall_ceiling@")))
    present, gold_counts = merged[f"{scorers[0]}/gold_in_pool"].astype(np.int64), merged[f"{scorers[0]}/gold_total"].astype(np.int64)
    ceiling = dict(headroom_metrics(present, gold_counts, ks=ks))
    ceiling.update(full_coverage_ceiling(present, gold_counts, ks=ks))
    perfect = ceiling.get("recall_ceiling_perfect_retrieval@5")
    if perfect:
        ceiling["fraction_of_attainable@5"] = float(ceiling["recall_ceiling@5"] / perfect)
    ceiling.update({"candidates_mean": float(sizes.mean()), "candidates_p50": float(np.percentile(sizes, 50)),
                    "candidates_p95": float(np.percentile(sizes, 95)), "candidates_max": int(sizes.max())})
    weighted = sum(r["ceiling_as_compiled"]["recall_ceiling@5"] * r["queries"] for r in records) / full_n
    if abs(weighted - ceiling["recall_ceiling@5"]) > 1e-9:
        raise SystemExit(f"{name}: merged recall_ceiling@5 {ceiling['recall_ceiling@5']} != shard-weighted {weighted}")
    audits = audit_scorers(merged, scorers, name + " (merged)")
    half = merged["half"].astype(bool)
    first = by_k[0]
    record = {"dataset": first["dataset"], "utc": utc(), "queries": int(full_n), "zero_gold_excluded": first["zero_gold_excluded"], "ids_sha256": full_digest,
              "supplement": first.get("supplement"), "shard": None, "merged_from": [{"k": r["shard"]["k"], "N": N, "queries": r["queries"], "utc": r["utc"], "seconds": r["seconds"],
                                              "peak_rss_bytes": r["peak_rss_bytes"], "threads": r["threads"]} for r in records],
              "contract_block": first["contract_block"], "core_sha256": first["core_sha256"], "m3b_contract_block": first["m3b_contract_block"],
              "pool": first["pool"], "construction": first["construction"],
              "ceiling_as_compiled": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in ceiling.items()},
              "seeds_added_mean": float(sum(r["seeds_added_mean"] * r["queries"] for r in records) / full_n), "scorers": scorers,
              "chunk_queries": first["chunk_queries"], "halves": {**first["halves"], "V2_GATE": int(half.sum()), "V2_HELD_CONFIRMATION": int((~half).sum())},
              "seconds": round(sum(r["seconds"] for r in records), 1),
              "ms_per_query": round(sum(r["ms_per_query"] * r["queries"] for r in records) / full_n, 2),
              "latency": first["latency"], "latency_measured_on": f"shard 0 of {N}, first {LATENCY_QUERIES} of its queries",
              "peak_rss_bytes": max(r["peak_rss_bytes"] for r in records), "threads": first["threads"],
              "freeze_RECORD_SHA256": first["freeze_RECORD_SHA256"], "mrr_audit": audits,
              "m3b_fixed_rrf_agreement": {"per_shard": [r["m3b_fixed_rrf_agreement"] for r in records]},
              "summary_half": first["summary_half"], "summary_V2_GATE": gate_summary(merged, scorers, half)}
    np.savez_compressed(EVAL / f"{name}.npz", **merged)
    (EVAL / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (EVAL / f"{name}_query_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    log(f"   {name}: merged {N} shards, {full_n} queries; V2_GATE " + "; ".join(f"{s.split('__')[0]} R@5={v['recall@5']:.3f}" for s, v in record["summary_V2_GATE"].items()))
    return record


def stage_eval(cfg: dict, cfg_m3b: dict, cfg_h: dict, inputs: dict, datasets: list[str], chunk_nodes: int, shard=None, models_only=None,
               log=print) -> None:
    """order_of_operations 6: every fitted arm on the three eval populations; refused before selection.json (check_3).
    With models_only (order_of_operations 7): the named fit keys -- seeds 1-2 of a passing arm -- scored on the same
    populations into a supplement record <name>__more_<sha8>, after the gate exists and only for keys no record holds.
    Every record carries both halves labelled; nothing here reads a number."""
    selection = read_json(OUT / "selection.json")
    if selection is None:
        raise SystemExit("no selection.json: the eval populations are not scored before the selection is filed (check_3)")
    bad = [d for d in datasets if d not in PILOT]
    if bad:
        raise SystemExit(f"{bad}: the pilot scores the three declared populations only (measurement.populations)")
    m3b_compile = M3B_RUN.load_script("m3b_compile")
    m3b_contract = M3B_RUN.load_script("m3b_contract")
    contexts, handles, pkg, bank = open_contexts_v2(cfg_m3b, datasets, m3b_compile)
    models = load_models(inputs, selection, bank)
    log(f"eval: {len(models)} models: {sorted(models)}")
    tag = None
    if models_only:
        missing = [k for k in models_only if k not in models]
        if missing:
            raise SystemExit(f"{missing}: no fitted model (record + weights) under outputs/universal_v2/fits/")
        if not (OUT / "gate_record.json").exists():
            raise SystemExit("a supplement eval pass follows the gate (order_of_operations 7); no gate_record.json")
        tag = "more_" + hashlib.sha256(",".join(sorted(models_only)).encode("utf-8")).hexdigest()[:8]
    for name in datasets:
        base, supplement = name, None
        if models_only:
            seed0 = read_json(EVAL / f"{name}.json")
            if seed0 is None:
                raise SystemExit(f"{name}: the seed-0 eval record is scored first; a supplement pass adds models to an existing population record")
            held = {k for p in EVAL.glob(f"{name}__more_*.json") if not p.name.endswith("_query_ids.json") for k in read_json(p)["scorers"]}
            already = [k for k in models_only if k in seed0["scorers"] or k in held]
            if already:
                raise SystemExit(f"{name}: {already} already scored on this population; a model is scored once")
            base = f"{name}__{tag}"
            supplement = {"tag": tag, "models": sorted(models_only), "seed0_record_sha256": sha256_file(EVAL / f"{name}.json")}
        if (EVAL / f"{base}.json").exists():
            log(f"   {base}: eval record exists, not repeated")
            continue
        if shard is not None and (EVAL / f"{base}{M3B_RUN.shard_suffix(shard)}.json").exists():
            log(f"   {base}: shard {shard[0]} of {shard[1]} exists, not repeated")
            continue
        eval_dataset(name, cfg, cfg_m3b, cfg_h, inputs, models, contexts[name], handles[name], pkg, m3b_compile, m3b_contract, chunk_nodes, shard=shard,
                     only_keys=models_only, tag=tag, supplement=supplement, log=log)
        gc.collect()


# ── the pilot gate: V2_GATE only, read once ──────────────────────────────────


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = BOOTSTRAP["resamples"], seed: int = BOOTSTRAP["seed"], level: int = BOOTSTRAP["level"]) -> dict:
    """measurement.paired_procedures: the mean of a - b per query and its percentile interval over resamples of the queries."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = int(d.size)
    means = np.asarray([d[rng.integers(n, size=n)].mean() for _ in range(resamples)]) if n else np.zeros(1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return {"mean": round(float(d.mean()) if n else 0.0, 4), "low": round(float(lo), 4), "high": round(float(hi), 4), "n": n, "resamples": resamples, "seed": seed}


def gate_thresholds(cfg: dict) -> tuple[dict, dict, str]:
    """The gate cells as last amended (the latest dated amendment carrying pilot_gate_amended), else pilot_gate."""
    for key in reversed(dated_blocks(cfg, "amendment")):
        block = cfg[key] if isinstance(cfg[key], dict) else {}
        if isinstance(block.get("pilot_gate_amended"), dict):
            g = block["pilot_gate_amended"]
            return g["gnn_gate"], g["twin_gate"], key + ".pilot_gate_amended"
    return cfg["pilot_gate"]["gnn_gate"], cfg["pilot_gate"]["twin_gate"], "pilot_gate"


def gate_cells(gnn: dict, twin: dict) -> dict:
    """(dataset, metric, slice, threshold, paired reference) per family; a paired cell also needs its interval
    against the frozen reference to exclude zero on the positive side."""
    return {"gnn": [("metaqa", "hit@1", "all", float(gnn["metaqa"]["hit1_all_hops"]), "gat_universal_v1"),
                    ("metaqa", "hit@1", "3hop", float(gnn["metaqa"]["hit1_3hop"]), "gat_universal_v1"),
                    ("2wiki", "recall@5", "all", float(gnn["2wiki"]["recall5"]), None),
                    ("2wiki", "full_coverage@5", "all", float(gnn["2wiki"]["full_coverage5"]), None),
                    ("squad", "recall@5", "all", float(gnn["squad"]["recall5"]), None)],
            "twin": [("metaqa", "hit@1", "all", float(twin["metaqa"]["hit1_all_hops"]), "qls_u_sota_v1"),
                     ("2wiki", "recall@5", "all", float(twin["2wiki"]["recall5"]), None),
                     ("squad", "recall@5", "all", float(twin["squad"]["recall5"]), None)]}


def gate_rows(name: str) -> tuple[dict, np.ndarray]:
    """The V2_GATE rows of one eval record: every array is sliced by the half label as it is read; the held rows
    never leave this function (check_3: the gate never reads V2_HELD_CONFIRMATION)."""
    with np.load(EVAL / f"{name}.npz") as z:
        half = z["half"].astype(bool)
        rows = {k: z[k][half] for k in z.files if k != "half"}
    return rows, half


def m3b_gate_rows(name: str, half: np.ndarray) -> dict:
    """The frozen M3B arrays of the same population on the same V2_GATE rows (the query ids must agree exactly)."""
    ours = json.loads((EVAL / f"{name}_query_ids.json").read_text(encoding="utf-8"))
    theirs = json.loads((M3B_OUT / "eval" / f"{name}_query_ids.json").read_text(encoding="utf-8"))
    if ours != theirs:
        raise SystemExit(f"{name}: the v2 and M3B eval populations are not the same query list; refusing")
    with np.load(M3B_OUT / "eval" / f"{name}.npz") as z:
        return {k: z[k][half] for k in z.files if k != "pool_size"}


def slice_mask(rows: dict, slice_: str) -> np.ndarray:
    n = next(iter(rows.values())).shape[0]
    if slice_ == "all":
        return np.ones(n, dtype=bool)
    if slice_.endswith("hop"):
        return rows["hop"] == int(slice_[0])
    raise ValueError(slice_)


def evaluate_cells(key: str, cells: list, rows: dict, refs: dict) -> dict:
    out = {"key": key, "cells": [], "pass": True}
    for name, metric, slice_, threshold, paired in cells:
        if f"{key}/{metric}" not in rows[name]:
            raise SystemExit(f"{key}: not scored on {name}")
        mask = slice_mask(rows[name], slice_)
        a = rows[name][f"{key}/{metric}"][mask]
        value = float(a.mean()) if a.size else 0.0
        entry = {"dataset": name, "metric": metric, "slice": slice_, "queries": int(mask.sum()), "value": round(value, 4), "threshold": threshold,
                 "at_or_above_threshold": bool(value >= threshold - 1e-12)}
        holds = entry["at_or_above_threshold"]
        if paired:
            b = refs[name][f"{M3B_REFERENCES[paired]}/{metric}"][mask]
            entry["paired_vs"] = paired
            entry["interval"] = paired_bootstrap(a, b)
            entry["interval_excludes_zero_on_the_positive_side"] = bool(entry["interval"]["low"] > 0)
            holds = holds and entry["interval_excludes_zero_on_the_positive_side"]
        entry["holds"] = bool(holds)
        out["cells"].append(entry)
        out["pass"] = out["pass"] and bool(holds)
    return out


def paired_table(rows: dict, refs: dict, pairs: list, metrics=("recall@5", "hit@1", "full_coverage@5")) -> dict:
    """measurement.paired_procedures.against_each_other and against_m3b, per dataset, on the rows given."""
    out: dict = {}
    for label, left, right, right_is_m3b in pairs:
        out[label] = {}
        for name in PILOT:
            r = rows[name]
            src = refs[name] if right_is_m3b else r
            if f"{left}/recall@5" not in r or f"{right}/recall@5" not in src:
                out[label][name] = None
                continue
            out[label][name] = {m: paired_bootstrap(r[f"{left}/{m}"], src[f"{right}/{m}"]) for m in metrics}
    return out


def scorer_keys(rows: dict) -> list[str]:
    return sorted({k.split("/")[0] for k in rows if k.endswith("/recall@5")})


def slices_on(rows: dict) -> dict:
    """measurement.slices_reported on the rows given: metaqa by hop, recall@5 by the gold's STRUCT distance
    from the seeds, the multi-gold queries."""
    out: dict = {"metaqa_by_hop": {}, "gold_distance_recall5": {}, "multi_gold": {}}
    labels = {0: "dist0", 1: "dist1", 2: "dist2", 3: "dist3", 4: "unreached"}
    for name in PILOT:
        r = rows[name]
        keys = scorer_keys(r)
        if name == "metaqa":
            for h in (1, 2, 3):
                m = r["hop"] == h
                out["metaqa_by_hop"][f"{h}hop"] = {"queries": int(m.sum()), **{k: {"hit@1": round(float(r[f"{k}/hit@1"][m].mean()), 4),
                                                                                   "recall@5": round(float(r[f"{k}/recall@5"][m].mean()), 4)} for k in keys}}
        dist = r["gold_dist_struct"]
        out["gold_distance_recall5"][name] = {}
        for d, label in labels.items():
            m = dist == d
            if m.any():
                out["gold_distance_recall5"][name][label] = {"queries": int(m.sum()), **{k: round(float(r[f"{k}/recall@5"][m].mean()), 4) for k in keys}}
        multi = r[f"{keys[0]}/gold_in_pool"] >= 2
        out["multi_gold"][name] = {"queries": int(multi.sum()), **{k: {"full_coverage@5": round(float(r[f"{k}/full_coverage@5"][multi].mean()), 4) if multi.any() else None,
                                                                       "recall@5": round(float(r[f"{k}/recall@5"][multi].mean()), 4) if multi.any() else None} for k in keys}}
    return out


def mechanism_on(rows: dict) -> dict:
    """measurement.mechanism_readouts on the rows given: the step / evidence / block gates (mean and quartiles)
    and the delta_s magnitude per arm per dataset."""
    out: dict = {}
    for name in PILOT:
        r = rows[name]
        out[name] = {}
        for k in scorer_keys(r):
            if k.startswith("fixed:"):
                continue
            entry: dict = {}
            for mname in sorted(a.split("/")[1] for a in r if a.startswith(k + "/") and (a.endswith("_ratio") or "gate" in a or a.endswith("changed"))):
                v = r[f"{k}/{mname}"]
                if mname in ("delta_ratio", "top1_changed"):
                    entry[mname] = round(float(v.mean()), 4)
                else:
                    entry[mname] = {"mean": round(float(v.mean()), 4), "q25": round(float(np.percentile(v, 25)), 4),
                                    "q50": round(float(np.percentile(v, 50)), 4), "q75": round(float(np.percentile(v, 75)), 4)}
            out[name][k] = entry
    return out


def family_outcome(selection: dict, outcome: dict) -> dict:
    """amendment 2 family_status_vocabulary: GNN_GATE and TWIN_GATE each PASS / FAIL for the selected arm of the
    family, and overall = BOTH_PASS | GNN_ONLY_PASS | TWIN_ONLY_PASS | PILOT_FAILED. A one-family pass is never
    described as a pass of the proposed universal pair."""
    passes = {fam: outcome[selection[fam]] == "PASS" for fam in ("gnn", "twin")}
    return {FAMILY_GATES["gnn"]: "PASS" if passes["gnn"] else "FAIL", FAMILY_GATES["twin"]: "PASS" if passes["twin"] else "FAIL",
            "overall": OVERALL[(passes["gnn"], passes["twin"])],
            "selected": {FAMILY_GATES[f]: selection[f] for f in ("gnn", "twin")}}


def terminal_state(family: dict, confirmation: dict) -> dict:
    """amendment 3 terminal_state_vocabulary: per family GATE_FAIL | CONFIRMATION_FAIL | CONFIRMED_PASS from the gate
    record's family_outcome and the report's seed confirmation (pilot_gate.on_pass: the mean over seeds 0-2 holds every
    cell); terminal = BOTH_CONFIRMED | GNN_ONLY_CONFIRMED | TWIN_ONLY_CONFIRMED | PILOT_FAILED. A family that passed at
    seed 0 without a complete confirmation refuses (the run record follows the confirmation seeds)."""
    final, confirmed = {}, {}
    for fam in ("gnn", "twin"):
        gate_name = FAMILY_GATES[fam]
        arm = family["selected"][gate_name]
        if family[gate_name] != "PASS":
            final[gate_name] = "GATE_FAIL"
        else:
            conf = confirmation.get(arm)
            if conf is None or not conf.get("complete"):
                raise SystemExit(f"{gate_name} {arm}: passed at seed 0 but its seed confirmation is incomplete; the run record follows seeds 1-2 (pilot_gate.on_pass)")
            final[gate_name] = "CONFIRMED_PASS" if conf["confirmed"] else "CONFIRMATION_FAIL"
        confirmed[fam] = final[gate_name] == "CONFIRMED_PASS"
    return {"family_final": final, "terminal": TERMINAL[(confirmed["gnn"], confirmed["twin"])], "selected": dict(family["selected"]),
            "gate_overall": family["overall"], "rule": "amendment_3_2026_09_19 terminal_state_vocabulary"}


def file_hard_stop(cfg: dict, date: str, condition: int, reason: str, evidence: dict, remaining_work: list, config: Path = CONFIG, log=print) -> str:
    """amendment 3 hard_stop_procedure: the dated block with the condition, the evidence, the exact remaining work and the
    state of every sidecar; the status line is not moved."""
    if condition not in range(1, 12):
        raise SystemExit(f"hard stop condition {condition}: the ruling names conditions 1-11")
    state = {}
    for p in sorted(OUT.rglob("*")) if OUT.exists() else []:
        if p.is_file() and p.suffix in (".json", ".yaml", ".pt", ".npz") and "cache" not in p.parts:
            state[p.relative_to(OUT).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    block = {"status": HARD_STOP, "filed_utc": utc(), "condition": condition, "reason": reason, "evidence": evidence,
             "remaining_work": remaining_work, "sidecars": state, "status_line": yaml.safe_load(config.read_text(encoding="utf-8"))["status"],
             "rule": "amendment_3_2026_09_19 hard_stop_procedure -- nothing further runs; the status line is not moved"}
    key = f"hard_stop_{date}"
    append_block(cfg, key, block, f"hard stop (amendment 3), condition {condition}, filed {block['filed_utc']}", config)
    log(f"filed {key}: condition {condition} -- {reason}")
    return key


def fit_hours_guard(arm: str, training: dict, log=print) -> dict:
    """amendment 3 compute_ceiling_guard: fit-hours spent (every fit record) plus the projected hours of this fit
    (timing.json: projected epoch minutes x max_epochs, scaled by the measured full-epoch / projected ratio of
    u_gnn_v2_ef) within compute.hard_ceiling 150 fit-hours; a breach is written and refuses the fit."""
    spent = 0.0
    for p in sorted(FITS.glob("*.json")) if FITS.exists() else []:
        spent += float(json.loads(p.read_text(encoding="utf-8")).get("seconds", 0.0)) / 3600
    timing = read_json(OUT / "timing.json")
    if timing is None:
        raise SystemExit("no timing.json: the timing run precedes every fit (order_of_operations 4)")
    ef = timing["arms"]["u_gnn_v2_ef"]
    ratio = float(timing["full_epoch_u_gnn_v2_ef"]["epoch_seconds"]) / max(60.0 * float(ef["projected_epoch_minutes"]), 1e-9)
    projected = float(timing["arms"][arm]["projected_epoch_minutes"]) * max(ratio, 1.0) * int(training["max_epochs"]) / 60
    out = {"arm": arm, "fit_hours_spent": round(spent, 3), "projected_fit_hours": round(projected, 3), "ratio_full_epoch_over_projected": round(ratio, 4),
           "ceiling_fit_hours": FIT_HOURS_CEILING, "within_ceiling": bool(spent + projected <= FIT_HOURS_CEILING)}
    if not out["within_ceiling"]:
        out["remaining_work"] = f"{arm}: {projected:.1f} projected fit-hours after {spent:.1f} spent; ceiling {FIT_HOURS_CEILING:.0f}"
        (OUT / "ceiling_breach.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        raise SystemExit(f"compute.hard_ceiling: {spent:.1f} fit-hours spent + {projected:.1f} projected for {arm} > {FIT_HOURS_CEILING:.0f}; not started (ceiling_breach.json)")
    log(f"   ceiling guard: {spent:.1f} fit-hours spent + {projected:.1f} projected for {arm} <= {FIT_HOURS_CEILING:.0f}")
    return out


# ── amendment 4: the post-pilot replication of the two selected arms ────────


def replication_amendment(cfg: dict) -> tuple[str, dict]:
    """The latest dated amendment carrying post_pilot_replication (amendment 4), or a refusal: a replication runs
    only under a block filed before its fits."""
    keys = [k for k in dated_blocks(cfg, "amendment") if isinstance(cfg[k], dict) and isinstance(cfg[k].get("post_pilot_replication"), dict)]
    if not keys:
        raise SystemExit("no dated amendment carries post_pilot_replication; the replication is filed before any of its fits (amendment 4)")
    return keys[-1], cfg[keys[-1]]["post_pilot_replication"]


def status_line(config: Path | None = None) -> str:
    """The one top-level status line as the file holds it (the loaded mapping may be stale in a long process)."""
    config = CONFIG if config is None else config   # resolved at call time (the tests re-point CONFIG)
    hits = [line for line in config.read_bytes().decode("utf-8").splitlines() if line.startswith("status: ")]
    if len(hits) != 1:
        raise SystemExit("the declaration must carry exactly one top-level status line")
    return hits[0][len("status: "):].strip()


def replication_authorisation(cfg: dict, arm: str, seed: int, config: Path | None = None) -> dict | None:
    """amendment 4 post_pilot_replication: (arm, seed) is fitted after the pilot closed only when the amendment names
    exactly that pair, the status line is the one it applies at, the pilot's run record is filed with the terminal state
    it cites, the arm is the selected arm of its family in selection.json, gate_record.json and the amendment, the pinned
    sidecars are unchanged and the seed-0 weights of the arm hash to what the run record pins. None when no amendment
    names the pair (the pilot_gate.on_pass rule then stands alone); a refusal when one does and a premise fails."""
    keys = [k for k in dated_blocks(cfg, "amendment") if isinstance(cfg[k], dict) and isinstance(cfg[k].get("post_pilot_replication"), dict)]
    if not keys:
        return None
    key, rep = keys[-1], cfg[keys[-1]]["post_pilot_replication"]
    if seed not in (rep.get("authorised_fits") or {}).get(arm, []):
        return None
    status = status_line(config)
    if status != rep["applies_at_status"]:
        raise SystemExit(f"{key}: applies at status {rep['applies_at_status']}; the declaration is at {status}")
    runs = dated_blocks(cfg, "run_record")
    if not runs:
        raise SystemExit(f"{key}: no run_record_<date> is filed; the replication follows the closed pilot")
    run = cfg[runs[-1]]
    terminal = (run.get("terminal_state") or {}).get("terminal")
    if terminal != rep["original_pilot_status"]:
        raise SystemExit(f"{key}: the run record closed at {terminal}, not the {rep['original_pilot_status']} the amendment cites")
    selection, gate = read_json(OUT / "selection.json"), read_json(OUT / "gate_record.json")
    if selection is None or gate is None:
        raise SystemExit(f"{key}: selection.json and gate_record.json are read; one is missing")
    family = FAMILY_OF_ARM[arm]
    if selection[family]["arm"] != arm or gate["selection"][family] != arm or rep["selected_arms"][family] != arm:
        raise SystemExit(f"{key}: {arm} is not the selected {family} arm of selection.json, gate_record.json and the amendment alike")
    pins = {}
    for name in ("selection.json", "gate_record.json", "held_record.json"):
        pinned = run.get(name.replace(".json", "") + "_sha256")
        if pinned and sha256_file(OUT / name) != pinned:
            raise SystemExit(f"{key}: {name} is not the file {runs[-1]} pins; nothing of the pilot is edited")
        pins[name] = pinned
    key0 = fit_key(arm, 0)
    seed0 = read_json(FITS / f"{key0}.json")
    if seed0 is None or not (FITS / f"{key0}.pt").exists():
        raise SystemExit(f"{key}: no seed-0 fit of {arm}; the replication adds seeds to a fitted arm, seed 0 is never retrained")
    pinned0 = ((run.get("fits") or {}).get(key0) or {}).get("state_sha256")
    if pinned0 != seed0["state_sha256"] or sha256_file(FITS / f"{key0}.pt") != pinned0:
        raise SystemExit(f"{key}: the seed-0 weights of {arm} do not hash to what {runs[-1]} pins; refusing")
    return {"block": key, "arm": arm, "seed": seed, "status_line": status, "run_record_block": runs[-1], "seed0_state_sha256": pinned0,
            "pins_verified": pins, "seed0_record": seed0}


def stage_gate(cfg: dict, inputs: dict, log=print) -> dict:
    """pilot_gate as amended: read once, on V2_GATE only, for the selected GNN and the selected twin; the
    non-selected candidates are reported and cannot advance; the paired tables, slices and mechanism readouts
    of the same half beside it. Refused before selection.json; refused a second time; never reads the held half."""
    selection = read_json(OUT / "selection.json")
    if selection is None:
        raise SystemExit("no selection.json: the gate is read for the selected GNN and the selected twin only (check_3)")
    if (OUT / "gate_record.json").exists():
        raise SystemExit("gate_record.json exists; the gate is read once (pilot_gate)")
    for name in PILOT:
        if not (EVAL / f"{name}.json").exists() or not (EVAL / f"{name}.npz").exists():
            raise SystemExit(f"{name}: no eval record; the gate needs the three pilot populations")
    gnn_thr, twin_thr, source = gate_thresholds(cfg)
    cells = gate_cells(gnn_thr, twin_thr)
    rows, refs = {}, {}
    for name in PILOT:
        rows[name], half = gate_rows(name)
        refs[name] = m3b_gate_rows(name, half)
    selected = {"gnn": selection["gnn"]["arm"], "twin": selection["twin"]["arm"]}
    verdict = {arm: evaluate_cells(fit_key(arm, 0), cells[family], rows, refs) for family, arm in selected.items()}
    reported = {}
    for family, candidates in (("gnn", GNN_CANDIDATES), ("twin", TWIN_CANDIDATES)):
        for arm in candidates:
            if arm not in verdict and f"{fit_key(arm, 0)}/recall@5" in rows["metaqa"]:
                reported[arm] = {**evaluate_cells(fit_key(arm, 0), cells[family], rows, refs), "cannot_advance": True,
                                 "why": "not the selected candidate of its family (arms.selection_behind_the_firewall)"}
    g, t = fit_key(selected["gnn"], 0), fit_key(selected["twin"], 0)
    pairs = [("selected_gnn_minus_selected_twin", g, t, False), ("selected_gnn_minus_u_gnn_v2_core78", g, fit_key("u_gnn_v2_core78", 0), False),
             ("selected_gnn_minus_gat_universal_v1_trio", g, fit_key("gat_universal_v1_trio", 0), False),
             ("u_gnn_v2_core78_minus_gat_universal_v1_trio", fit_key("u_gnn_v2_core78", 0), fit_key("gat_universal_v1_trio", 0), False),
             ("selected_gnn_minus_gat_universal_v1_frozen", g, M3B_REFERENCES["gat_universal_v1"], True),
             ("selected_twin_minus_qls_u_sota_v1_frozen", t, M3B_REFERENCES["qls_u_sota_v1"], True),
             ("selected_twin_minus_gat_no_mp_v1_frozen", t, M3B_REFERENCES["gat_no_mp_v1"], True)]
    record = {"utc": utc(), "half": "V2_GATE", "held_half_read": False, "thresholds_from": source, "selection": selected,
              "selection_sha256": sha256_file(OUT / "selection.json"), "contract_block": inputs["contract_block"], "core_sha256": inputs["core_sha256"],
              "verdict": verdict, "outcome": {arm: ("PASS" if v["pass"] else "FAIL") for arm, v in verdict.items()},
              "family_outcome": family_outcome(selected, {arm: ("PASS" if v["pass"] else "FAIL") for arm, v in verdict.items()}),
              "reported_not_advancing": reported, "paired_on_V2_GATE": paired_table(rows, refs, pairs),
              "frozen_references_on_V2_GATE": {name: {label: {m: round(float(refs[name][f"{key}/{m}"].mean()), 4) for m in ("recall@5", "hit@1", "full_coverage@5")}
                                                      for label, key in M3B_REFERENCES.items()} for name in PILOT},
              "queries_V2_GATE": {name: int(next(iter(rows[name].values())).shape[0]) for name in PILOT},
              "slices_V2_GATE": slices_on(rows), "mechanism_readouts_V2_GATE": mechanism_on(rows), "bootstrap": BOOTSTRAP,
              "eval_records": {name: {"record_sha256": sha256_file(EVAL / f"{name}.json"), "arrays_sha256": sha256_file(EVAL / f"{name}.npz")} for name in PILOT},
              "m3b_reference_arrays": {name: sha256_file(M3B_OUT / "eval" / f"{name}.npz") for name in PILOT},
              "on_pass": str(cfg["pilot_gate"]["on_pass"]).strip(), "on_fail": str(cfg["pilot_gate"]["on_fail"]).strip()}
    (OUT / "gate_record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    for arm, v in verdict.items():
        log(f"gate {arm}: {record['outcome'][arm]}; " + "; ".join(f"{c['dataset']}/{c['metric']}/{c['slice']} {c['value']} vs {c['threshold']} {'holds' if c['holds'] else 'FAILS'}" for c in v["cells"]))
    fam = record["family_outcome"]
    log(f"gate families: GNN_GATE {fam['GNN_GATE']}, TWIN_GATE {fam['TWIN_GATE']}, overall {fam['overall']}")
    return record


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["compile", "screen", "file", "timing", "fit", "select", "eval", "merge", "gate"], required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--kinds", nargs="*", default=["fit", "select"], help="compile: the carves to cache (fit, select)")
    parser.add_argument("--which", nargs="*", default=None, help="file: contract, timing, gate, run")
    parser.add_argument("--date", default=None, help="file: the date suffix of the appended block, e.g. 2026_09_20")
    parser.add_argument("--arm", default=None, help="fit: one of the declared arms")
    parser.add_argument("--seed", type=int, default=0, help="fit: 0, or 1-2 for an arm that passed its gate")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pack-workers", type=int, default=PACK["workers"], help="threads packing batches ahead of the step")
    parser.add_argument("--prefetch-depth", type=int, default=PACK["depth"], help="batches packed ahead")
    parser.add_argument("--chunk-nodes", type=int, default=24000, help="eval: candidate rows packed per forward pass")
    parser.add_argument("--shard", default=None, help="eval: k/N scores queries k::N of the population; --stage merge joins the shards")
    parser.add_argument("--models", nargs="*", default=None, help="eval: fit keys of a supplement pass (seeds 1-2 after the gate)")
    args = parser.parse_args()
    shard = None
    if args.shard:
        k, N = (int(v) for v in args.shard.split("/"))
        if not 0 <= k < N:
            raise SystemExit(f"--shard {args.shard}: need 0 <= k < N")
        shard = (k, N)
    sys.dont_write_bytecode = True
    torch.set_num_threads(args.threads)
    PACK["workers"], PACK["depth"] = max(1, args.pack_workers), max(1, args.prefetch_depth)
    cfg, cfg_m3b, cfg_h = load_configs()
    datasets = args.datasets or list(PILOT)
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"_run_{args.stage}{'' if shard is None else f'_shard{shard[0]}of{shard[1]}'}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + LF)

    log(f"stage {args.stage}: M3B pins verified; threads {torch.get_num_threads()}, pack workers {PACK['workers']} x depth {PACK['depth']}")
    t0 = time.time()
    if args.stage == "compile":
        stage_compile(cfg, cfg_m3b, cfg_h, datasets, tuple(args.kinds), log=log)
    elif args.stage == "screen":
        stage_screen(cfg, cfg_m3b, datasets, log=log)
    elif args.stage == "file":
        if not args.date or not args.which:
            raise SystemExit("--stage file needs --date and --which")
        stage_file(cfg, args.date, list(args.which), log=log)
    else:
        inputs = model_inputs(cfg, cfg_m3b)
        log(f"   {inputs['contract_block']}: {inputs['n_scalars']} columns ({inputs['core_sha256'][:12]}), base {inputs['base']}, "
            f"evidence {inputs['evidence']} (substitutions {inputs['evidence_substitutions']})")
        if args.stage == "eval":
            stage_eval(cfg, cfg_m3b, cfg_h, inputs, datasets, args.chunk_nodes, shard=shard, models_only=args.models, log=log)
        elif args.stage == "merge":
            for name in datasets:   # a dataset, or a supplement base <dataset>__more_<sha8>
                if (EVAL / f"{name}.json").exists():
                    log(f"   {name}: eval record exists, not merged again")
                else:
                    merge_shards(name, log=log)
        elif args.stage == "gate":
            stage_gate(cfg, inputs, log=log)
        elif args.stage == "select":
            stage_select(cfg, inputs, log=log)
        else:
            training = training_rule_v2(cfg, cfg_m3b)
            m3b_compile = M3B_RUN.load_script("m3b_compile")
            contexts, _, _, bank = open_contexts_v2(cfg_m3b, list(PILOT), m3b_compile)
            carves = open_carves_v2(contexts, inputs)
            log(f"   training rule: {training}; relation bank {bank.n_rows} rows ({bank.sha256[:12]})")
            if args.stage == "timing":
                stage_timing(cfg, inputs, carves, bank, training, log=log)
            else:
                if not args.arm:
                    raise SystemExit("--stage fit needs --arm")
                run_fit(args.arm, int(args.seed), inputs, carves, bank, training, log=log, cfg=cfg)
    log(f"stage {args.stage}: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
