#!/usr/bin/env python
"""M2's own runner: the QLS-UNIVERSAL screen, one seed, fully instrumented.

Reads configs/m2_qls_v2_freeze.yaml#m2_selection_matrix -- M1A's declaration is
never consulted for what to run -- and fits exactly the arms that matrix marks
``new``, plus any arm scripts/m2_reuse_audit.py refused (the declaration's own
on_failure rule: a refused reuse becomes NEW, it is not argued back into reuse).

Why a separate runner rather than a flag on M1A's
--------------------------------------------------
1. run_m1a_feature_screen.run() reads its cells from M1A's frozen declaration
   and its --arms flag can only narrow them, so it cannot run an arm M1A never
   declared -- and the universal arm must not be added to M1A's matrix
   (tests/test_m2_universal_schema.py asserts it appears in no M1A cell).
2. Its main() hard-fails on seed != 0 and its _run_arm discards the per-query
   rows M2's instrumentation requires and saves no checkpoint.
3. Editing it would break the code-identity premise scripts/m2_reuse_audit.py's
   bit-exact probe depends on.

So the feature path is *imported and called unmodified* -- _cell_master_local,
_widen_query, _arm_store from M1A; _fit, _score_once from the confirmation
trainer -- and only the orchestration and the recording are new. That is the
same relationship run_m1b_targeted_resolution.py has to M1A, for the same
reason.

Instrumentation is not optional here
------------------------------------
Every fit persists the fifteen fields
configs/m2_qls_v2_freeze.yaml#instrumentation_requirement.exact_fields_every_
new_m2_fit_writes names, including a checkpoint, the per-query rows, and the
feature store it was trained on. The runner re-aggregates the rows it just
wrote and refuses to record a fit whose aggregate does not reproduce -- M1B's
amendment 5 had to re-run 27 completed fits for exactly this, and a check that
runs at write time is what makes it impossible to repeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").is_file():  # pragma: no cover - Modal layout
    REPO_ROOT = next(
        (Path(entry) for entry in sys.path
         if entry and (Path(entry) / "configs" / "m2_qls_v2_freeze.yaml").is_file()),
        REPO_ROOT,
    )
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.graph_context import build_operators  # noqa: E402
from mp_retrieval.headroom_v2 import ragged_from_rows, regime_headroom  # noqa: E402
from mp_retrieval.structural_features import StructuralFeatureStore  # noqa: E402
from scripts import run_m1a_feature_screen as _m1a  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d1 import holdout_split  # noqa: E402
from scripts.run_m0a_probe import (  # noqa: E402
    QueryView,
    _families,
    _load_family_csr,
    _peak_rss_bytes,
    _percentiles,
    _undirected,
)
from scripts.run_m0b_regime_map import MAINLINE_FAMILY, _a64_budget  # noqa: E402
from scripts.run_operator_screen import _aggregate_rows  # noqa: E402
from scripts.run_sa_mlp_confirmation import _fit, _score_once, validate_candidate_contract  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
REUSE_MANIFEST_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "reuse_audit.json"
COMPLETE_STATUS = "M2_QLS_V2_FREEZE_DATASET_COMPLETE"
BUILD_COMPLETE_STATUS = "M2_QLS_V2_FREEZE_FEATURE_BUILD_COMPLETE"
KS = _m1a.KS

#: How much of a dataset one invocation does.
#:
#: ``full``  -- build the features and fit, in one container. The path M1A and
#:              M1B used, and still the default.
#: ``build`` -- build and persist every declared cell's master block, then stop.
#:              Runs on a CPU container: the build receives only numpy arrays
#:              and plain objects (proved by
#:              scripts/m2_feature_build_equivalence.py stage 1), so it cannot
#:              observe whether an accelerator is attached, and holding an idle
#:              A10G through it is 83% of M2's estimated GPU bill.
#: ``fit``   -- load those persisted masters and fit, on the GPU container.
#:
#: ``build`` and ``fit`` write to the SAME artifact root by design: the fit
#: stage reads exactly the masters the build stage wrote, under the same
#: <artifact_root>/<regime>/cell_features path the ``full`` path already used.
STAGES = ("full", "build", "fit")

#: The declaration writes the universal arm by its scientific name; the runner's
#: arm vocabulary writes it by its family composition. One mapping, in one place.
DECLARED_UNIVERSAL_ARM = "QLS-UNIVERSAL"
RUNNER_UNIVERSAL_ARM = _m1a.UNIVERSAL_ARM

#: M2 is a one-seed screen. Extra seeds need their own amendment
#: (standing_prohibitions_restated.no_extra_seeds_beyond_seed_zero...).
DECLARED_SEED = 0

#: Written alongside every persisted store so a reader can tell which code
#: produced it without re-deriving that from a path.
FEATURE_STORE_FORMAT = "fixed_structural_features_v1"


def _load_declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _config_sha256() -> str:
    return hashlib.sha256(DECLARATION_PATH.read_bytes()).hexdigest()


def _source_commit(explicit: str | None) -> str | None:
    """The commit this fit ran at.

    Passed in by the launcher, because a Modal container has the source but not
    the .git directory. Resolved locally when it was not passed, and recorded as
    null rather than guessed when neither is available -- a wrong commit is
    worse than an absent one.
    """

    if explicit:
        return explicit
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return None


def runner_arm(declared: str) -> str:
    return RUNNER_UNIVERSAL_ARM if declared == DECLARED_UNIVERSAL_ARM else declared


def arm_slug(declared: str) -> str:
    return declared.replace("+", "_plus_").replace("-", "_").lower()


def declared_cells(declaration: dict[str, Any], dataset: str) -> dict[str, dict[str, str]]:
    """This dataset's (regime -> {declared arm: matrix status}) from M2's own matrix."""

    cells = declaration["m2_selection_matrix"]["cells"]
    if dataset not in cells:
        raise ValueError(
            f"{dataset!r} is not in M2's selection matrix; declared datasets are {sorted(cells)}"
        )
    for regime, arms in cells[dataset].items():
        unknown = sorted(
            arm for arm in arms if runner_arm(arm) not in _m1a.ARM_FAMILIES
        )
        if unknown:
            raise ValueError(
                f"Declared arm(s) {unknown} for {dataset!r}/{regime} have no family composition"
            )
    return cells[dataset]


def load_reuse_manifest() -> dict[tuple[str, str, str], bool]:
    """Which proposed reused fits the audit actually cleared.

    Refused fits come back False and are then run as new, which is the
    declaration's own on_failure rule. A missing or failed manifest stops the
    runner rather than letting it guess.
    """

    if not REUSE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"{REUSE_MANIFEST_PATH} does not exist -- run scripts/m2_reuse_audit.py "
            "before this runner; reuse must be proven per fit, never assumed"
        )
    manifest = json.loads(REUSE_MANIFEST_PATH.read_text(encoding="utf-8"))
    probe = manifest.get("bit_exact_feature_probe", {})
    if not probe.get("all_historical_arms_bit_exact"):
        raise ValueError(
            f"{REUSE_MANIFEST_PATH}: the bit-exact historical-arm probe did not pass "
            f"({probe.get('status')!r}). No M2 fit may run against a feature path whose "
            "equivalence to the one every reused comparator used is unproven."
        )
    return {(f["dataset"], f["regime"], f["arm"]): bool(f["reusable"]) for f in manifest["fits"]}


def arms_to_fit(
    cells: dict[str, dict[str, str]],
    dataset: str,
    reuse_lookup: dict[tuple[str, str, str], bool],
) -> dict[str, list[str]]:
    """Per regime, the declared arms this runner must actually fit.

    ``new`` always. A ``reuse_*`` arm only when the audit refused it -- then it
    is NEW by the declaration's own rule, and the workload counts the audit
    emitted already reflect it.
    """

    plan: dict[str, list[str]] = {}
    for regime, arms in cells.items():
        selected = []
        for arm, status in arms.items():
            if status == "new":
                selected.append(arm)
            elif str(status).startswith("reuse_"):
                if not reuse_lookup.get((dataset, regime, runner_arm(arm)), False):
                    selected.append(arm)
            else:
                raise ValueError(f"Unknown matrix status {status!r} for {dataset}/{regime}/{arm}")
        plan[regime] = selected
    return plan


# --- feature-store persistence ------------------------------------------------


def _sha256_of_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def save_feature_store(store: StructuralFeatureStore, root: Path, *, extra: dict[str, Any]) -> str:
    """Write the store in the existing on-disk format and return its fingerprint.

    ``fixed_structural_features_v1``, unchanged, so StructuralFeatureStore.load
    reads it back with no new reader. Returned fingerprint covers exactly the
    arrays the trainer consumes -- the float16 local block, the candidate
    pointer and the query positions -- so two stores with the same fingerprint
    train identically by construction.
    """

    root.mkdir(parents=True, exist_ok=True)
    local = np.ascontiguousarray(store.local)
    candidate_ptr = np.ascontiguousarray(store.candidate_ptr)
    query_position = np.ascontiguousarray(store.query_position)
    static = np.ascontiguousarray(store.static)
    np.save(root / "local.npy", local)
    np.save(root / "candidate_ptr.npy", candidate_ptr)
    np.save(root / "query_position.npy", query_position)
    np.save(root / "static.npy", static)
    fingerprint = _sha256_of_arrays(local, candidate_ptr, query_position)
    metadata = dict(store.metadata)
    metadata.update(extra)
    metadata["format"] = FEATURE_STORE_FORMAT
    metadata["persisted"] = True
    metadata["fingerprint_sha256"] = fingerprint
    metadata["local_shape"] = [int(local.shape[0]), int(local.shape[1])]
    metadata["local_dtype"] = str(local.dtype)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return fingerprint


def cell_build_key(args: argparse.Namespace, regime: str) -> dict[str, Any]:
    """Everything that decides what a cell's master block IS.

    The fit stage loads a master block some other container built. A hash of
    that block proves it was not corrupted in transit; it says nothing about
    whether it is a block of the RIGHT cell. This key is the input side: the
    dataset and its fingerprint, the panel, the A64 budget, the family. The fit
    stage recomputes it and refuses a store whose key differs -- loudly, rather
    than rebuilding silently, because a mismatch means the wrong build was run
    and quietly papering over that is how a phase reports a delta between two
    different experiments.
    """

    return {
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "regime": regime,
        "queries": int(args.queries),
        "per_seed_cap": int(args.per_seed_cap),
        "neighbour_scan_cap_per_seed": int(args.neighbour_scan_cap_per_seed),
        "a64_mainline_family": args.a64_mainline_family if regime == "R3" else None,
        "config_sha256": _config_sha256(),
    }


def save_cell_features(
    root: Path,
    scored_sets: list[np.ndarray],
    master_blocks: list[np.ndarray],
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """The per-cell master block, which is the expensive object.

    Persisted separately from the per-arm stores because it is what a CPU
    container hands to a GPU trainer under feature_build_compute_check:
    _cell_master_local is 83% of M2's estimated compute and every arm in the
    cell slices this one result. Ragged, so stored flat with an offset vector.

    ``extra`` carries the build key and the feature-build latencies. The
    latencies are persisted because the instrumentation requirement asks every
    fit to record p50/p95/p99 of the build it used, and under the split the fit
    stage did not perform that build -- measuring the load instead would put a
    different quantity under the same name.
    """

    root.mkdir(parents=True, exist_ok=True)
    scored = np.concatenate(scored_sets) if scored_sets else np.zeros(0, dtype=np.int64)
    master = np.concatenate(master_blocks, axis=0) if master_blocks else np.zeros((0, 12), np.float32)
    offsets = np.zeros(len(scored_sets) + 1, dtype=np.int64)
    for index, block in enumerate(scored_sets):
        offsets[index + 1] = offsets[index] + block.size
    np.save(root / "scored_flat.npy", scored)
    np.save(root / "scored_offsets.npy", offsets)
    np.save(root / "master_flat.npy", master)
    fingerprint = _sha256_of_arrays(scored, offsets, master)
    metadata = {
        "format": "m2_cell_master_features_v1",
        "built_by": "scripts/run_m2_qls_v2_freeze.py",
        "queries": len(scored_sets),
        "master_columns": int(master.shape[1]) if master.size else 0,
        "master_dtype": str(master.dtype),
        "fingerprint_sha256": fingerprint,
        "why_persisted": (
            "_cell_master_local dominates M2's cost and every arm in the cell "
            "slices this one result; persisting it is what makes a CPU-build / "
            "GPU-fit split an orchestration change rather than a rewrite"
        ),
    }
    metadata.update(extra or {})
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return fingerprint


def load_cell_features(root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read back what save_cell_features wrote, exactly."""

    scored = np.load(root / "scored_flat.npy")
    offsets = np.load(root / "scored_offsets.npy")
    master = np.load(root / "master_flat.npy")
    scored_sets, master_blocks, cursor = [], [], 0
    for index in range(offsets.size - 1):
        size = int(offsets[index + 1] - offsets[index])
        scored_sets.append(scored[int(offsets[index]): int(offsets[index + 1])])
        master_blocks.append(master[cursor: cursor + size])
        cursor += size
    return scored_sets, master_blocks


def load_cell_metadata(root: Path) -> dict[str, Any]:
    return json.loads((root / "metadata.json").read_text(encoding="utf-8"))


def load_cell_for_fit(
    root: Path, args: argparse.Namespace, regime: str
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float], str]:
    """A persisted cell, checked against the key the fit stage expects.

    Three refusals, each of which would otherwise be a silently wrong result:
    a missing store (the build stage was never run for this cell), a key that
    does not match (the wrong build was run), and a fingerprint that does not
    match the arrays on disk (the store was corrupted or hand-edited).
    """

    if not (root / "metadata.json").is_file():
        raise FileNotFoundError(
            f"{root} holds no persisted cell master -- run --stage build for {args.dataset}/"
            f"{regime} before --stage fit, or use --stage full to do both in one container"
        )
    metadata = load_cell_metadata(root)
    expected = cell_build_key(args, regime)
    recorded = metadata.get("build_key")
    if recorded != expected:
        differing = sorted(
            key for key in set(expected) | set(recorded or {})
            if (recorded or {}).get(key) != expected.get(key)
        )
        raise ValueError(
            f"{root}: the persisted cell master was built for a different cell -- {differing} "
            f"differ (persisted {recorded!r}, expected {expected!r}). Refusing rather than "
            "rebuilding silently: a key mismatch means the wrong build was run."
        )
    scored_sets, master_blocks = load_cell_features(root)
    fingerprint = _sha256_of_arrays(
        np.concatenate(scored_sets) if scored_sets else np.zeros(0, dtype=np.int64),
        np.load(root / "scored_offsets.npy"),
        np.concatenate(master_blocks, axis=0) if master_blocks
        else np.zeros((0, 12), dtype=np.float32),
    )
    if fingerprint != metadata.get("fingerprint_sha256"):
        raise ValueError(
            f"{root}: the arrays on disk do not hash to the fingerprint recorded beside them "
            f"({fingerprint} != {metadata.get('fingerprint_sha256')})"
        )
    latency = metadata.get("uncached_feature_build_latency_ms")
    if not latency:
        raise ValueError(
            f"{root}: no feature-build latency was persisted, so a fit loading this store could "
            "not record the p50/p95/p99 instrumentation_requirement asks for"
        )
    return scored_sets, master_blocks, latency, fingerprint


# --- one fit ------------------------------------------------------------------


def fit_one_arm(
    *,
    declared_arm: str,
    regime: str,
    store: StructuralFeatureStore,
    precomputed_width: int,
    train_queries: list[Any],
    validation_queries: list[Any],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    fit_root: Path,
    provenance: dict[str, Any],
    feature_build_latency_ms: dict[str, float],
    feature_store_fingerprint: str,
) -> dict[str, Any]:
    """Fit, score, and write everything the instrumentation requirement names.

    The body up to ``_score_once`` is M1A's ``_run_arm`` unchanged; what is new
    is that the rows ``_score_once`` already computed are kept instead of
    discarded, and that the fit is not recorded until re-aggregating those rows
    reproduces the aggregate it would otherwise have reported alone.
    """

    arm = runner_arm(declared_arm)
    _m1a.seed_everything(args.seed)
    model = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=args.semantic_rung,
        dropout=args.dropout,
        temperature=args.temperature,
        embedding_dim=node_embeddings.shape[1],
    )
    fitted, training = _fit(
        "sa_mlp", model, train_queries, validation_queries,
        node_embeddings, query_embeddings, None, store, device,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        seed=args.seed,
    )
    metrics, rows, inference = _score_once(
        "sa_mlp", fitted, validation_queries, node_embeddings, query_embeddings,
        None, store, device, batch_size=args.batch_size, ks=KS, timed=True,
    )

    if not rows:
        raise ValueError(
            f"{args.dataset}/{regime}/{declared_arm}: scoring produced no per-query rows, so "
            "nothing could reproduce the aggregate -- refusing to record a fit that cannot be "
            "audited afterwards"
        )
    query_ids = [query.query_id for query in validation_queries]
    reconstructed = _aggregate_rows(rows)
    mismatched = _aggregate_mismatch(metrics, reconstructed)
    if mismatched:
        raise ValueError(
            f"{args.dataset}/{regime}/{declared_arm}: re-aggregating the stored per-query rows "
            f"does not reproduce the reported metrics for {mismatched} -- the rows would not be "
            "sufficient to reproduce R@1/R@5/R@20/MRR/FullCov@20, which is the one thing "
            "instrumentation_requirement exists to guarantee"
        )

    fit_root.mkdir(parents=True, exist_ok=True)
    torch.save(fitted.state_dict(), fit_root / "checkpoint.pt")
    _atomic_json(
        fit_root / "per_query_rows.json",
        {
            "dataset": args.dataset,
            "regime": regime,
            "arm": declared_arm,
            "seed": args.seed,
            "query_ids": query_ids,
            "rows": rows,
            "aggregate_from_rows": reconstructed,
            "row_key_order": list(rows[0]) if rows else [],
        },
    )
    store_fingerprint = save_feature_store(
        store,
        fit_root / "feature_store",
        extra={
            "dataset": args.dataset,
            "regime": regime,
            "declared_arm": declared_arm,
            "runner_arm": arm,
            "precomputed_width": precomputed_width,
            "built_by": "scripts/run_m2_qls_v2_freeze.py",
        },
    )
    if store_fingerprint != feature_store_fingerprint:  # pragma: no cover - defensive
        raise ValueError("persisted feature store does not match the one the fit trained on")

    parameters = {
        "total": fitted.trainable_parameter_count(),
        "semantic": fitted.semantic_parameter_count(),
        "scorer": fitted.scorer_parameter_count(),
    }
    return {
        "arm": declared_arm,
        "runner_arm": arm,
        "regime": regime,
        "seed": args.seed,
        "semantic_rung": args.semantic_rung,
        "precomputed_width": precomputed_width,
        "parameters": parameters,
        "metrics": metrics,
        "training": training,
        "inference": inference,
        "eligible_train_queries": sum(1 for query in train_queries if query.relevant_local.numel()),
        "systems": {
            "train_time_seconds": training["training_seconds"],
            "peak_train_vram_mb": training["peak_training_gpu_memory_mb_total"],
            "peak_train_rss_mb": inference["peak_cpu_rss_mb_total"],
            "peak_train_rss_mb_provenance": (
                "measured during post-fit validation scoring, not the training loop itself -- "
                "_fit is imported unmodified and does not instrument CPU RSS"
            ),
        },
        "instrumentation": {
            "checkpoint": str((fit_root / "checkpoint.pt").relative_to(fit_root.parents[2]))
            if len(fit_root.parents) > 2 else str(fit_root / "checkpoint.pt"),
            "per_query_rows": str(fit_root / "per_query_rows.json"),
            "query_ids": len(query_ids),
            "candidate_ids_sha256": provenance["candidate_id_order_sha256"],
            "aggregate_metrics_reconstructed_from_rows": True,
            "source_commit": provenance["source_commit"],
            "config_sha256": provenance["config_sha256"],
            "dataset_fingerprint_sha256": provenance["dataset_fingerprint_sha256"],
            "candidate_contract_sha256": provenance["candidate_contract_sha256"],
            "feature_store_fingerprint_sha256": store_fingerprint,
            "training_seconds": training["training_seconds"],
            "feature_build_latency_ms_p50_p95_p99": {
                "p50": feature_build_latency_ms["p50"],
                "p95": feature_build_latency_ms["p95"],
                "p99": feature_build_latency_ms["p99"],
            },
            "peak_gpu_memory_mb": {
                "training_total": training["peak_training_gpu_memory_mb_total"],
                "inference_total": inference["peak_gpu_memory_mb_total"],
                "inference_incremental": inference["peak_gpu_memory_mb_incremental"],
            },
            "peak_cpu_rss_mb": {
                "total": inference["peak_cpu_rss_mb_total"],
                "incremental": inference["peak_cpu_rss_mb_incremental"],
            },
            "parameter_counts_total_semantic_scorer": parameters,
        },
    }


def _aggregate_mismatch(reported: dict[str, Any], reconstructed: dict[str, Any]) -> list[str]:
    """Which metrics the stored rows fail to reproduce.

    Compared exactly, not within a tolerance: the rows are the same objects the
    reported aggregate was built from, so anything but equality means they were
    not stored faithfully. NaN-vs-NaN counts as agreement (an all-None
    conditional column aggregates to NaN in both).
    """

    mismatched = []
    for key, value in reported.items():
        other = reconstructed.get(key)
        if isinstance(value, float) and isinstance(other, float):
            if np.isnan(value) and np.isnan(other):
                continue
        if other != value:
            mismatched.append(key)
    return mismatched


# --- the run ------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage = getattr(args, "stage", "full") or "full"
    if stage not in STAGES:
        raise ValueError(f"--stage must be one of {STAGES}; got {stage!r}")
    expected_status = BUILD_COMPLETE_STATUS if stage == "build" else COMPLETE_STATUS
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == expected_status:
            return existing
    if args.seed != DECLARED_SEED:
        raise ValueError(
            f"M2 is a seed-{DECLARED_SEED} screen; --seed {args.seed} needs its own amendment "
            "(standing_prohibitions_restated.no_extra_seeds_beyond_seed_zero_without_a_further_amendment)"
        )

    declaration = _load_declaration()
    cells = declared_cells(declaration, args.dataset)
    reuse_lookup = load_reuse_manifest()
    plan = arms_to_fit(cells, args.dataset, reuse_lookup)
    if args.regimes is not None:
        unknown = sorted(set(args.regimes) - set(plan))
        if unknown:
            raise ValueError(f"--regimes {unknown} are not declared for {args.dataset!r}: {sorted(plan)}")
        plan = {regime: plan[regime] for regime in args.regimes}
    if args.arms is not None:
        plan = {regime: [arm for arm in arms if arm in args.arms] for regime, arms in plan.items()}
    plan = {regime: arms for regime, arms in plan.items() if arms}
    if not plan:
        raise ValueError(
            f"nothing to fit for {args.dataset!r} under regimes={args.regimes} arms={args.arms} -- "
            "every selected arm is already covered by an audited reuse"
        )

    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=True)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    if dataset.node_array.shape[1] != args.frozen_embedding_dim:
        raise ValueError(
            f"{args.dataset}: node embeddings are {dataset.node_array.shape[1]}-dimensional, "
            f"--frozen-embedding-dim says {args.frozen_embedding_dim} -- SemanticHead would "
            "silently build at the wrong width and every downstream parameter count would be wrong"
        )
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(queries) < args.queries:
        raise ValueError(f"validation split holds {len(queries)} queries, needed {args.queries}")
    views = [QueryView(query) for query in queries]
    num_nodes = int(dataset.num_nodes)
    query_count = len(dataset.queries)

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    # Each half of the split pays only for what it uses. The build stage never
    # trains, so it never materialises the embedding tables (1536 floats per
    # node is the largest thing in the process); the fit stage never builds, so
    # it never pays for the CSR operators or the A64 family graph.
    operators = dense = splade = None
    family_rowptr = family_col = None
    device = node_embeddings = query_embeddings = None
    if stage != "fit":
        operators = build_operators(rowptr, col, num_nodes)
        dense = np.load(Path(args.data) / "dense_top200_all.npy", mmap_mode="r")
        splade = np.load(Path(args.data) / "splade_top200_all.npy", mmap_mode="r")
        if "R3" in plan:
            families = _families(args)
            if args.a64_mainline_family not in families:
                raise ValueError(
                    f"declared mainline family {args.a64_mainline_family!r} is not among "
                    f"{list(families)}"
                )
            family_rowptr, family_col = _load_family_csr(
                families[args.a64_mainline_family], num_nodes
            )
            family_rowptr, family_col, _symmetric = _undirected(family_rowptr, family_col, num_nodes)
    if stage != "build":
        device = (
            torch.device(args.device) if args.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        node_embeddings = torch.from_numpy(
            np.array(dataset.node_array, dtype=np.float32, copy=True)
        ).to(device)
        query_embeddings = torch.from_numpy(
            np.array(dataset.query_array, dtype=np.float32, copy=True)
        ).to(device)
    budget = _a64_budget(
        per_seed_cap=args.per_seed_cap, neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed
    )

    provenance = {
        "source_commit": _source_commit(args.source_commit),
        "config_sha256": _config_sha256(),
        "dataset_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract_sha256": candidate_contract["observed_contract_sha256"],
        "candidate_id_order_sha256": candidate_contract["candidate_id_order_sha256"],
    }
    artifact_root = args.artifact_root or (args.output.parent / "fits")

    golds = ragged_from_rows([view.golds for view in views])
    result_cells: dict[str, Any] = {}
    for regime, declared_arms in plan.items():
        cell_root = artifact_root / regime
        if stage == "fit":
            scored_sets, master_blocks, latency, cell_fingerprint = load_cell_for_fit(
                cell_root / "cell_features", args, regime
            )
        else:
            scored_sets, master_blocks, latencies = _m1a._cell_master_local(
                regime=regime,
                views=views,
                queries=queries,
                dense=dense,
                splade=splade,
                rowptr=rowptr,
                col=col,
                num_nodes=num_nodes,
                operators=operators,
                family_rowptr=family_rowptr,
                family_col=family_col,
                node_embeddings=dataset.node_array,
                budget=budget,
            )
            latency = _percentiles(latencies)
            cell_fingerprint = save_cell_features(
                cell_root / "cell_features",
                scored_sets,
                master_blocks,
                extra={
                    "build_key": cell_build_key(args, regime),
                    "uncached_feature_build_latency_ms": latency,
                    "built_at_stage": stage,
                    "source_commit": _source_commit(args.source_commit),
                },
            )
        headroom, _present, _gold_counts = regime_headroom(
            ragged_from_rows(scored_sets), golds, num_nodes=num_nodes, ks=KS
        )
        widened = [
            _m1a._widen_query(query, scored) for query, scored in zip(queries, scored_sets, strict=True)
        ]
        train_queries, held_out_queries = holdout_split(widened, args.holdout_fraction)

        cell_result: dict[str, Any] = {
            "regime": regime,
            "regime_headroom": headroom,
            "scored_node_count": _percentiles([float(scored.size) for scored in scored_sets]),
            "uncached_feature_build_latency_ms": latency,
            "train_queries": len(train_queries),
            "held_out_queries": len(held_out_queries),
            "arms_run": list(declared_arms),
            "cell_features_fingerprint_sha256": cell_fingerprint,
            "cell_features_root": str(cell_root / "cell_features"),
            "feature_build_stage": stage,
            "arms": {},
        }
        if stage == "build":
            # The whole point of this stage: the expensive object is written and
            # the container exits without ever needing an accelerator.
            result_cells[regime] = cell_result
            continue
        for declared_arm in declared_arms:
            arm = runner_arm(declared_arm)
            store, precomputed_width = _m1a._arm_store(
                arm=arm, regime=regime, master_blocks=master_blocks, queries=widened,
                query_count=query_count, num_nodes=num_nodes,
            )
            fingerprint = _sha256_of_arrays(
                np.ascontiguousarray(store.local),
                np.ascontiguousarray(store.candidate_ptr),
                np.ascontiguousarray(store.query_position),
            )
            arm_result = fit_one_arm(
                declared_arm=declared_arm,
                regime=regime,
                store=store,
                precomputed_width=precomputed_width,
                train_queries=train_queries,
                validation_queries=held_out_queries,
                node_embeddings=node_embeddings,
                query_embeddings=query_embeddings,
                device=device,
                args=args,
                fit_root=cell_root / arm_slug(declared_arm),
                provenance=provenance,
                feature_build_latency_ms=latency,
                feature_store_fingerprint=fingerprint,
            )
            achieved = arm_result["metrics"].get("recall@5")
            ceiling = headroom.get("recall_ceiling@5")
            arm_result["ceiling_attainment_at_5"] = (
                float(achieved) / float(ceiling) if achieved is not None and ceiling else None
            )
            arm_result["matrix_status"] = cells[regime][declared_arm]
            cell_result["arms"][declared_arm] = arm_result
        result_cells[regime] = cell_result

    result: dict[str, Any] = {
        "status": BUILD_COMPLETE_STATUS if stage == "build" else COMPLETE_STATUS,
        "stage": "m2_qls_v2_freeze",
        "feature_build_stage": stage,
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "candidate_contract": candidate_contract,
        "queries": len(views),
        "split": "validation",
        "selection": "deterministic_prefix_of_the_split_order",
        "test_split_read": False,
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "semantic_rung": args.semantic_rung,
        "num_nodes": num_nodes,
        "a64_mainline_family": args.a64_mainline_family if "R3" in plan else None,
        "provenance": provenance,
        "artifact_root": str(artifact_root),
        "cells": result_cells,
        "systems": {"peak_process_rss_bytes": _peak_rss_bytes()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--frozen-embedding-dim", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--per-seed-cap", type=int, default=16)
    parser.add_argument("--neighbour-scan-cap-per-seed", type=int, default=4096)
    parser.add_argument("--edge-provenance-root", type=Path, default=None)
    parser.add_argument("--edge-families", nargs="+", default=[MAINLINE_FAMILY])
    parser.add_argument("--a64-mainline-family", default=MAINLINE_FAMILY)
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--semantic-rung", default="S3", choices=("S2", "S3"))
    parser.add_argument("--seed", type=int, default=DECLARED_SEED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument(
        "--stage",
        default="full",
        choices=STAGES,
        help=(
            "full: build and fit in one container (the proven path). "
            "build: persist every declared cell's master block and stop, so the build can run "
            "off the GPU. fit: load those masters and fit. build and fit share --artifact-root."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed != DECLARED_SEED:
        parser.error(
            f"M2 declares seed {DECLARED_SEED} only; --seed {args.seed} needs a further amendment"
        )
    args.baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
