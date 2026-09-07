"""M2B's runner: the same frozen model with a different semantic branch.

Everything about a fit is M2's -- the cell, the candidate set, the context, the
structural columns, the scorer shape, the loss, the hyperparameters, the seed,
the holdout boundary -- except which semantic head is passed in. That is the
whole experimental design, and it is enforced structurally rather than by care:
the structural block is not rebuilt here at all but loaded from the cell master
M2 already persisted, and the rungs of one cell are handed the same
``StructuralFeatureStore`` object. A difference between S2, S3 and S4 in a cell
is therefore a difference in semantic representation or it is nothing.

Because "structurally guaranteed" is an argument and not an observation, the
cell loop also computes a ``shared_inputs_sha256`` over everything that must
not vary -- the persisted cell master, the arm store, the held-out query IDs,
the precomputed width, the candidate contract and normalisation, the per-family
structural column digests -- and refuses to record a cell whose rungs disagree
on it.

Three things this runner does that M2's does not.

It loads a store by its scientific contract instead of by the hash of a
declaration file. M2's ``load_cell_for_fit`` compares the SHA-256 of the whole
YAML, which M2B's own amendment already invalidated without changing a single
float. ``feature_build_contract`` replaces that with a hash over only what can
change a cell tensor, and this runner recomputes the candidate contract in the
container so the data side of that hash is measured here rather than inherited.

It benchmarks the whole served path, not the scorer. S4's two ``1536 -> 64``
projections are the expensive thing under test; timing an already-computed
semantic tensor would charge them nothing.

It re-benchmarks S3 rather than reusing M2's timings. S3's fitted weights and
its effectiveness metrics are reused because the fit is the same fit; a latency
measured on another day in another container is a measurement of that
container, and the systems tie-break orders on latency.

One container per dataset, several fits inside it, each independently
checkpointed and independently addressable at
``<artifact_root>/<regime>/<rung>/``. Sharing a container saves startup, not
identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.headroom_v2 import ragged_from_rows, regime_headroom  # noqa: E402
from mp_retrieval.m2b_inference_timing import (  # noqa: E402
    DEFAULT_REPEATS,
    DEFAULT_WARMUP_QUERIES,
    measure_uncached_inference,
)
from mp_retrieval.m2b_semantic_control import ProjectionSemanticHead  # noqa: E402
from mp_retrieval.qls_v2_semantic import SemanticHead  # noqa: E402
from scripts import feature_build_contract as fbc  # noqa: E402
from scripts import run_m1a_feature_screen as _m1a  # noqa: E402
from scripts import run_m2_qls_v2_freeze as _m2  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d1 import holdout_split  # noqa: E402
from scripts.run_operator_screen import _aggregate_rows  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _fit,
    _score_once,
    validate_candidate_contract,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"

STATUS_COMPLETE = "M2B_SEMANTIC_MINIMALITY_DATASET_COMPLETE"
KS = _m1a.KS

#: The one arm M2 selected, under the name M2 filed it as. The runner-side
#: spelling is ``_m1a.UNIVERSAL_ARM``; both travel with every fit so a reader
#: never has to guess which vocabulary a field is written in.
DECLARED_UNIVERSAL_ARM = _m2.DECLARED_UNIVERSAL_ARM
RUNNER_UNIVERSAL_ARM = _m2.RUNNER_UNIVERSAL_ARM

#: S3's fits already exist. S2 and S4 are the new work; S3 appears here only to
#: be re-benchmarked for inference from its reused checkpoint.
NEW_RUNGS = ("S2", "S4")
REUSED_RUNG = "S3"
ALL_RUNGS = ("S2", "S3", "S4")

DECLARED_SEED = 0

#: The width every parameter count in this phase is quoted at. Named here so the
#: declared-parameter check fires on the real thing and stays quiet on a toy
#: fixture, which legitimately weighs something else.
FROZEN_EMBEDDING_DIM = 1536

#: What each rung weighs at the frozen width, measured live during the
#: reconnaissance and re-asserted at every fit. A rung that silently changed
#: width would otherwise produce a perfectly plausible result under the wrong
#: name, and the whole phase is a statement about parameter counts.
DECLARED_PARAMETERS: dict[str, dict[str, int]] = {
    "S2": {"semantic": 0, "total": 449, "semantic_columns": 3},
    "S3": {"semantic": 3072, "total": 3585, "semantic_columns": 5},
    "S4": {"semantic": 196608, "total": 205217, "semantic_columns": 258},
}


def _sha256_of_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_semantic_head(rung: str, dim: int) -> torch.nn.Module:
    """The one place a rung name becomes a module."""

    if rung in ("S2", "S3"):
        return SemanticHead(rung=rung, dim=dim)
    if rung == "S4":
        return ProjectionSemanticHead(dim=dim)
    raise ValueError(f"unknown semantic rung {rung!r}; M2B declares {list(ALL_RUNGS)}")


def semantic_rung_fingerprint(head: torch.nn.Module) -> dict[str, Any]:
    """What distinguishes two fits that differ only in semantic representation.

    Without it an S2 fit and an S4 fit in the same cell are nearly
    indistinguishable in their provenance -- same dataset, same regime, same
    store, same seed, same commit -- which is precisely the confusion a phase
    that varies one branch is capable of producing.
    """

    identity: dict[str, Any] = {
        "rung": head.rung,
        "module": type(head).__module__,
        "class": type(head).__name__,
        "feature_names": list(head.feature_names),
        "semantic_columns": len(head.feature_names),
        "embedding_dim": int(head.dim),
        "projection_dim": int(getattr(head, "projection_dim", 0)) or None,
        "semantic_parameters": int(sum(p.numel() for p in head.parameters())),
    }
    identity["sha256"] = _sha256_of_json(identity)
    return identity


def declared_cells(dataset: str) -> list[str]:
    """This dataset's regimes, read from M2B's matrix rather than from a flag."""

    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    cells = declaration["evaluation_matrix"]["cells"]
    if dataset not in cells:
        raise ValueError(
            f"{dataset!r} is not in M2B's evaluation matrix; declared datasets are "
            f"{sorted(cells)}"
        )
    return list(cells[dataset])


def structural_family_digests(master_blocks: list[np.ndarray]) -> dict[str, str]:
    """A digest per structural family, so "NODE_ROLE is identical" is checkable.

    The smoke has to establish that S2 and S4 saw the same NODE_ROLE, SUPPORT
    and PATH values. "They shared a tensor" is an argument; these digests are
    the observation, and they are also what a later phase would compare against
    M2's own stores.
    """

    if not master_blocks:
        return {family: _m2._sha256_of_arrays() for family in _m1a.MASTER_COLUMNS}
    master = np.concatenate(master_blocks, axis=0)
    return {
        family: _m2._sha256_of_arrays(np.ascontiguousarray(master[:, columns]))
        for family, columns in _m1a.MASTER_COLUMNS.items()
    }


# --- loading a cell M2 built -----------------------------------------------------


def load_cell_under_contract(
    root: Path,
    args: argparse.Namespace,
    regime: str,
    *,
    candidate_contract: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float], str, dict[str, Any]]:
    """A persisted M2 cell master, admitted on its scientific identity.

    Refuses for the three reasons M2's loader refuses -- no store, wrong cell,
    arrays that do not hash to their recorded fingerprint -- and for one it
    could not: a formula constant that has moved since the store was built. It
    does NOT refuse for a changed declaration hash, which is the whole point.
    """

    if not (root / "metadata.json").is_file():
        raise FileNotFoundError(
            f"{root} holds no persisted cell master. M2B never builds features; it reads "
            f"the master M2 wrote for {args.dataset}/{regime}."
        )
    metadata = _m2.load_cell_metadata(root)

    forward = fbc.contract_from_runner_args(
        args,
        regime,
        candidate_contract_sha256=candidate_contract["observed_contract_sha256"],
        candidate_id_order_sha256=candidate_contract["candidate_id_order_sha256"],
        store_format=metadata["format"],
        master_columns=int(metadata["master_columns"]),
        master_dtype=str(metadata["master_dtype"]),
    )
    expected_key = {
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "regime": regime,
        "queries": int(args.queries),
        "per_seed_cap": int(args.per_seed_cap),
        "neighbour_scan_cap_per_seed": int(args.neighbour_scan_cap_per_seed),
        "a64_mainline_family": args.a64_mainline_family if regime == "R3" else None,
    }
    decision = fbc.verify_store_identity(
        store_metadata=metadata,
        expected_build_key=expected_key,
        forward_contract=forward,
    )
    if not decision["admitted"]:
        raise ValueError(f"{root}: {decision['why']}")

    scored_sets, master_blocks = _m2.load_cell_features(root)
    fingerprint = _m2._sha256_of_arrays(
        np.concatenate(scored_sets) if scored_sets else np.zeros(0, dtype=np.int64),
        np.load(root / "scored_offsets.npy"),
        np.concatenate(master_blocks, axis=0) if master_blocks
        else np.zeros((0, 12), dtype=np.float32),
    )
    if fingerprint != metadata.get("fingerprint_sha256"):
        raise ValueError(
            f"{root}: the arrays on disk do not hash to the fingerprint recorded beside "
            f"them ({fingerprint} != {metadata.get('fingerprint_sha256')})"
        )
    latency = metadata.get("uncached_feature_build_latency_ms")
    if not latency:
        raise ValueError(
            f"{root}: no feature-build latency was persisted, so a fit loading this store "
            "could not record the p50/p95/p99 the instrumentation requirement asks for"
        )
    return scored_sets, master_blocks, latency, fingerprint, decision


# --- what a fit has to survive before it is recorded -----------------------------


def _one_query_scores(
    model: torch.nn.Module,
    query: Any,
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    store: Any,
    device: torch.device,
) -> torch.Tensor:
    candidate_index = query.candidate_index.to(device)
    nodes = node_embeddings[candidate_index]
    query_row = query_embeddings[
        torch.tensor([query.query_index], dtype=torch.long, device=device)
    ]
    structural = store.batch_features(
        [query], include_static=True, include_local=True, device=device
    )
    batch_index = torch.zeros(nodes.shape[0], dtype=torch.long, device=device)
    return model.forward_explicit(nodes, query_row, batch_index, structural)


def finite_scores(
    model: torch.nn.Module,
    queries: list[Any],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    store: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Every held-out score, checked for NaN and Inf.

    The aggregate metrics can look perfectly healthy while individual scores
    are not: a rank is an argsort, and argsorting a vector that contains NaN
    still returns a permutation.
    """

    was_training = model.training
    model.eval()
    minimum, maximum, counted = float("inf"), -float("inf"), 0
    try:
        with torch.no_grad():
            for query in queries:
                scores = _one_query_scores(
                    model, query, node_embeddings, query_embeddings, store, device
                )
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError(
                        f"query {query.query_id} produced a non-finite score. The aggregate "
                        "metrics would still have looked healthy, because ranking a vector "
                        "containing NaN still returns a permutation."
                    )
                minimum = min(minimum, float(scores.min()))
                maximum = max(maximum, float(scores.max()))
                counted += int(scores.numel())
    finally:
        model.train(was_training)
    return {
        "scored_values": counted,
        "min": minimum if counted else None,
        "max": maximum if counted else None,
        "all_finite": True,
    }


def checkpoint_round_trips(
    checkpoint: Path,
    model: torch.nn.Module,
    *,
    rung: str,
    precomputed_width: int,
    query: Any,
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    store: Any,
    device: torch.device,
    dropout: float,
    temperature: float,
) -> dict[str, Any]:
    """The saved checkpoint rebuilds this exact model, proved by forward equality.

    "A checkpoint was written" is a file-existence claim. What the phase needs
    -- and what any later reuse of these weights depends on, exactly as M2B
    itself reuses M2's S3 checkpoints -- is that loading it back reproduces the
    scores this fit reported.
    """

    embedding_dim = int(node_embeddings.shape[1])
    reloaded = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=rung,
        dropout=dropout,
        temperature=temperature,
        embedding_dim=embedding_dim,
        semantic_head=build_semantic_head(rung, embedding_dim),
    ).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )

    was_training = model.training
    model.eval()
    reloaded.eval()
    try:
        with torch.no_grad():
            original = _one_query_scores(
                model, query, node_embeddings, query_embeddings, store, device
            )
            restored = _one_query_scores(
                reloaded, query, node_embeddings, query_embeddings, store, device
            )
    finally:
        model.train(was_training)
    difference = float((original - restored).abs().max())
    if difference != 0.0:
        raise ValueError(
            f"reloading {checkpoint} does not reproduce this fit's scores (max absolute "
            f"difference {difference}); the checkpoint is not the model that was measured"
        )
    return {
        "loaded_strict": True,
        "max_absolute_score_difference": difference,
        "compared_on_query": query.query_id,
    }


def check_declared_parameters(rung: str, parameters: dict[str, int], columns: int) -> None:
    expected = DECLARED_PARAMETERS[rung]
    observed = {
        "semantic": int(parameters["semantic"]),
        "total": int(parameters["total"]),
        "semantic_columns": int(columns),
    }
    if observed != expected:
        raise ValueError(
            f"{rung} weighs {observed} but M2B's declaration says {expected}. Refusing: the "
            "parameter count is how this phase's claim is stated, and a fit recorded under "
            "the wrong one is unrecoverable after the fact."
        )


# --- one fit ---------------------------------------------------------------------


def reused_checkpoint_path(args: argparse.Namespace, regime: str) -> Path:
    """Where M2 wrote the fit whose weights S3 reuses.

    Refuses rather than returning ``None`` when no root was given. A ``None``
    would fall through to the training branch, and this phase would then report
    a fresh S3 fit in a column its own declaration calls reused -- a fabricated
    reuse that nothing downstream could detect.
    """

    if not args.m2_fits_root:
        raise ValueError(
            f"{REUSED_RUNG} is a reused rung and no --m2-fits-root was given, so there is "
            "nothing to reuse. Refusing rather than refitting silently: the declaration's "
            "workload says 14 fits are reused, and a fresh fit reported in that column "
            "would be a fabricated reuse."
        )
    return (
        Path(args.m2_fits_root) / regime / _m2.arm_slug(DECLARED_UNIVERSAL_ARM) / "checkpoint.pt"
    )


def fit_one_rung(
    *,
    rung: str,
    regime: str,
    store: Any,
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
    shared_inputs: dict[str, Any],
    reused_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """One (cell, rung) fit, or one reused fit re-benchmarked.

    ``reused_checkpoint`` is how S3 enters M2B: its weights and its
    effectiveness are M2's, so nothing is trained, but the model is rebuilt
    here and scored here, so its metrics come from this container's own rows
    and its latency from this container's own clock.
    """

    _m1a.seed_everything(args.seed)
    embedding_dim = int(node_embeddings.shape[1])
    model = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=rung,
        dropout=args.dropout,
        temperature=args.temperature,
        embedding_dim=embedding_dim,
        semantic_head=build_semantic_head(rung, embedding_dim),
    ).to(device)

    if reused_checkpoint is not None:
        if not reused_checkpoint.is_file():
            raise FileNotFoundError(
                f"{reused_checkpoint} does not exist, so {rung} cannot be reused for "
                f"{args.dataset}/{regime}. Refusing rather than silently refitting: a refit "
                "would not be the fit whose effectiveness numbers M2 filed, and reporting it "
                "as one would be a fabricated reuse."
            )
        model.load_state_dict(
            torch.load(reused_checkpoint, map_location=device, weights_only=True), strict=True
        )
        fitted = model
        training = {
            "reused": True,
            "reused_from": str(reused_checkpoint),
            "training_seconds": 0.0,
            "peak_training_gpu_memory_mb_total": 0.0,
            "history": [],
            "why_zero": (
                "no training happened here. M2 paid for this fit; M2B loads its weights so "
                "the effectiveness numbers are the same fit's, and re-measures only what is "
                "a property of this container -- the latency the tie-break orders on."
            ),
        }
    else:
        fitted, training = _fit(
            "sa_mlp", model, train_queries, validation_queries,
            node_embeddings, query_embeddings, None, store, device,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            seed=args.seed,
        )
        training["reused"] = False
        losses = [epoch["loss"] for epoch in training["history"]]
        if not losses or not all(bool(np.isfinite(loss)) for loss in losses):
            raise ValueError(
                f"{args.dataset}/{regime}/{rung}: training losses {losses} are not all "
                "finite, so the fit diverged and its metrics mean nothing"
            )

    metrics, rows, inference = _score_once(
        "sa_mlp", fitted, validation_queries, node_embeddings, query_embeddings,
        None, store, device, batch_size=args.batch_size, ks=KS, timed=True,
    )
    if not rows:
        raise ValueError(
            f"{args.dataset}/{regime}/{rung}: scoring produced no per-query rows, so nothing "
            "could reproduce the aggregate"
        )
    reconstructed = _aggregate_rows(rows)
    mismatched = _m2._aggregate_mismatch(metrics, reconstructed)
    if mismatched:
        raise ValueError(
            f"{args.dataset}/{regime}/{rung}: re-aggregating the stored per-query rows does "
            f"not reproduce the reported metrics for {mismatched}"
        )
    non_finite = sorted(
        name for name, value in metrics.items()
        if isinstance(value, (int, float)) and not bool(np.isfinite(float(value)))
    )
    if non_finite:
        raise ValueError(f"{args.dataset}/{regime}/{rung}: metrics {non_finite} are not finite")

    scores = finite_scores(
        fitted, validation_queries, node_embeddings, query_embeddings, store, device
    )

    parameters = {
        "total": fitted.trainable_parameter_count(),
        "semantic": fitted.semantic_parameter_count(),
        "scorer": fitted.scorer_parameter_count(),
    }
    fingerprint = semantic_rung_fingerprint(fitted.semantic_head)
    if fingerprint["semantic_parameters"] != parameters["semantic"]:  # pragma: no cover
        raise ValueError("the rung fingerprint disagrees with the model it fingerprints")
    if embedding_dim == FROZEN_EMBEDDING_DIM:
        # Only meaningful at the frozen width; a toy-width model built by a test
        # legitimately weighs something else.
        check_declared_parameters(rung, parameters, fingerprint["semantic_columns"])

    latency = measure_uncached_inference(
        model=fitted,
        queries=validation_queries[: args.latency_queries],
        node_embeddings=node_embeddings,
        query_embeddings=query_embeddings,
        store=store,
        device=device,
        repeats=args.latency_repeats,
        warmup=args.latency_warmup,
    )

    fit_root.mkdir(parents=True, exist_ok=True)
    checkpoint = fit_root / "checkpoint.pt"
    torch.save(fitted.state_dict(), checkpoint)
    round_trip = checkpoint_round_trips(
        checkpoint, fitted,
        rung=rung, precomputed_width=precomputed_width, query=validation_queries[0],
        node_embeddings=node_embeddings, query_embeddings=query_embeddings,
        store=store, device=device, dropout=args.dropout, temperature=args.temperature,
    )
    query_ids = [query.query_id for query in validation_queries]
    _atomic_json(
        fit_root / "per_query_rows.json",
        {
            "dataset": args.dataset,
            "regime": regime,
            "arm": DECLARED_UNIVERSAL_ARM,
            "semantic_rung": rung,
            "seed": args.seed,
            "query_ids": query_ids,
            "rows": rows,
            "aggregate_from_rows": reconstructed,
            "row_key_order": list(rows[0]),
        },
    )

    result = {
        "dataset": args.dataset,
        "regime": regime,
        "arm": DECLARED_UNIVERSAL_ARM,
        "runner_arm": RUNNER_UNIVERSAL_ARM,
        "semantic_rung": rung,
        "seed": args.seed,
        "reused_from_m2": reused_checkpoint is not None,
        "precomputed_width": precomputed_width,
        "semantic_rung_fingerprint": fingerprint,
        "shared_inputs_sha256": shared_inputs["sha256"],
        "parameters": parameters,
        "metrics": metrics,
        "training": training,
        "batched_inference": inference,
        "uncached_inference": latency,
        "held_out_scores": scores,
        "checkpoint_round_trip": round_trip,
        "eligible_train_queries": sum(
            1 for query in train_queries if query.relevant_local.numel()
        ),
        "systems": {
            "train_time_seconds": training["training_seconds"],
            "peak_train_vram_mb": training["peak_training_gpu_memory_mb_total"],
            "peak_inference_vram_mb": latency["peak_inference_gpu_memory_mb"],
            "peak_rss_mb": inference["peak_cpu_rss_mb_total"],
            "uncached_inference_p50_ms": latency["total_model_ms"]["p50"],
            "uncached_inference_p95_ms": latency["total_model_ms"]["p95"],
            "uncached_inference_p99_ms": latency["total_model_ms"]["p99"],
            "tie_break_orders_on": "uncached_inference_p95_ms",
        },
        "instrumentation": {
            "checkpoint": str(checkpoint),
            "per_query_rows": str(fit_root / "per_query_rows.json"),
            "query_ids": len(query_ids),
            "aggregate_metrics_reconstructed_from_rows": True,
            "source_commit": provenance["source_commit"],
            "m2b_declaration_sha256": provenance["config_sha256"],
            "dataset_fingerprint_sha256": provenance["dataset_fingerprint_sha256"],
            "candidate_contract_sha256": provenance["candidate_contract_sha256"],
            "candidate_ids_sha256": provenance["candidate_id_order_sha256"],
            "feature_store_fingerprint_sha256": shared_inputs["arm_store_sha256"],
            "cell_features_fingerprint_sha256": shared_inputs["cell_features_sha256"],
            "feature_build_contract_sha256": shared_inputs["feature_build_contract_sha256"],
            "semantic_rung_fingerprint_sha256": fingerprint["sha256"],
            "semantic_parameter_count": parameters["semantic"],
            "scorer_parameter_count": parameters["scorer"],
            "total_parameter_count": parameters["total"],
            "uncached_feature_build_latency_ms": feature_build_latency_ms,
            "peak_gpu_memory_mb": {
                "training_total": training["peak_training_gpu_memory_mb_total"],
                "uncached_inference_total": latency["peak_inference_gpu_memory_mb"],
                "batched_inference_total": inference["peak_gpu_memory_mb_total"],
            },
        },
    }
    # Every fit is addressable on its own, without the dataset artifact, because
    # 28 fits sharing 6 containers must not become 6 results.
    _atomic_json(fit_root / "fit.json", result)
    return result


# --- the dataset -------------------------------------------------------------------


def shared_inputs_of_cell(
    *,
    cell_features_sha256: str,
    arm_store_sha256: str,
    feature_build_contract_sha256: str,
    precomputed_width: int,
    query_ids: list[Any],
    family_digests: dict[str, str],
    candidate_contract: dict[str, Any],
) -> dict[str, Any]:
    """Everything that must be identical across the rungs of one cell.

    Recorded on every fit and compared afterwards, so "only the semantic rung
    differs" is something the artifact can fail rather than a sentence in a
    declaration.
    """

    shared = {
        "cell_features_sha256": cell_features_sha256,
        "arm_store_sha256": arm_store_sha256,
        "feature_build_contract_sha256": feature_build_contract_sha256,
        "precomputed_width": int(precomputed_width),
        "held_out_query_ids_sha256": _sha256_of_json([str(qid) for qid in query_ids]),
        "held_out_query_count": len(query_ids),
        "structural_family_digests": family_digests,
        "candidate_contract_sha256": candidate_contract["observed_contract_sha256"],
        "candidate_id_order_sha256": candidate_contract["candidate_id_order_sha256"],
        "candidate_normalisation": fbc.formula_identity()["feature_normalisation"],
    }
    shared["sha256"] = _sha256_of_json(shared)
    return shared


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == STATUS_COMPLETE:
            return existing
    if args.seed != DECLARED_SEED:
        raise ValueError(
            f"M2B is a seed-{DECLARED_SEED} screen; --seed {args.seed} needs its own "
            "amendment (seed_policy)"
        )
    available = declared_cells(args.dataset)
    regimes = list(args.regimes) if args.regimes else available
    unknown = sorted(set(regimes) - set(available))
    if unknown:
        raise ValueError(
            f"--regimes {unknown} are not declared for {args.dataset!r}: {available}"
        )
    rungs = list(args.rungs) if args.rungs else list(ALL_RUNGS)
    unknown_rungs = sorted(set(rungs) - set(ALL_RUNGS))
    if unknown_rungs:
        raise ValueError(f"--rungs {unknown_rungs} are not M2B's rungs {list(ALL_RUNGS)}")

    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=True)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    if dataset.node_array.shape[1] != args.frozen_embedding_dim:
        raise ValueError(
            f"{args.dataset}: node embeddings are {dataset.node_array.shape[1]}-dimensional, "
            f"--frozen-embedding-dim says {args.frozen_embedding_dim} -- every semantic head "
            "would build at the wrong width and every parameter count would be wrong"
        )
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(queries) < args.queries:
        raise ValueError(f"validation split holds {len(queries)} queries, needed {args.queries}")

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
    num_nodes = int(dataset.num_nodes)
    query_count = len(dataset.queries)

    provenance = {
        "source_commit": _m2._source_commit(args.source_commit),
        "config_sha256": hashlib.sha256(DECLARATION_PATH.read_bytes()).hexdigest(),
        "dataset_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract_sha256": candidate_contract["observed_contract_sha256"],
        "candidate_id_order_sha256": candidate_contract["candidate_id_order_sha256"],
    }
    artifact_root = Path(args.artifact_root or (args.output.parent / "fits"))
    # M2's fits are a sealed artifact. M2B reads their cell masters and writes
    # nowhere near them, so the two roots are separate arguments even though a
    # local test run points both at the same directory.
    cell_master_root = Path(
        args.cell_master_root if args.cell_master_root is not None else artifact_root
    )
    golds = ragged_from_rows([_m1a.QueryView(query).golds for query in queries])

    cells: dict[str, Any] = {}
    for regime in regimes:
        cell_root = artifact_root / regime
        cell_features = cell_master_root / regime / "cell_features"
        scored_sets, master_blocks, build_latency, cell_fingerprint, decision = (
            load_cell_under_contract(
                cell_features, args, regime, candidate_contract=candidate_contract,
            )
        )
        headroom, _present, _counts = regime_headroom(
            ragged_from_rows(scored_sets), golds, num_nodes=num_nodes, ks=KS
        )
        widened = [
            _m1a._widen_query(query, scored)
            for query, scored in zip(queries, scored_sets, strict=True)
        ]
        train_queries, held_out = holdout_split(widened, args.holdout_fraction)

        # One store, built once, handed to every rung in this cell. The rungs
        # cannot diverge on structure because there is only one structure.
        store, precomputed_width = _m1a._arm_store(
            arm=RUNNER_UNIVERSAL_ARM, regime=regime, master_blocks=master_blocks,
            queries=widened, query_count=query_count, num_nodes=num_nodes,
        )
        arm_store_sha256 = _m2.save_feature_store(
            store,
            cell_root / "arm_store",
            extra={
                "dataset": args.dataset,
                "regime": regime,
                "declared_arm": DECLARED_UNIVERSAL_ARM,
                "runner_arm": RUNNER_UNIVERSAL_ARM,
                "precomputed_width": precomputed_width,
                "built_by": "scripts/run_m2b_semantic_minimality.py",
                "shared_by_rungs": list(rungs),
            },
        )
        shared = shared_inputs_of_cell(
            cell_features_sha256=cell_fingerprint,
            arm_store_sha256=arm_store_sha256,
            feature_build_contract_sha256=decision["feature_build_contract_sha256"],
            precomputed_width=precomputed_width,
            query_ids=[query.query_id for query in held_out],
            family_digests=structural_family_digests(master_blocks),
            candidate_contract=candidate_contract,
        )

        cell: dict[str, Any] = {
            "regime": regime,
            "regime_headroom": headroom,
            "train_queries": len(train_queries),
            "held_out_queries": len(held_out),
            "precomputed_width": precomputed_width,
            "cell_features_root": str(cell_features),
            "arm_store_root": str(cell_root / "arm_store"),
            "store_admission": decision,
            "uncached_feature_build_latency_ms": build_latency,
            "shared_inputs": shared,
            "every_rung_reads_this_one_store": (
                "The structural block is loaded, never rebuilt, and one StructuralFeatureStore "
                "is handed to every rung in this cell. A difference between the rungs here is "
                "a difference in semantic representation or it is nothing."
            ),
            "rungs": {},
        }
        for rung in rungs:
            fit = fit_one_rung(
                rung=rung,
                regime=regime,
                store=store,
                precomputed_width=precomputed_width,
                train_queries=train_queries,
                validation_queries=held_out,
                node_embeddings=node_embeddings,
                query_embeddings=query_embeddings,
                device=device,
                args=args,
                fit_root=cell_root / rung.lower(),
                provenance=provenance,
                feature_build_latency_ms=build_latency,
                shared_inputs=shared,
                reused_checkpoint=(
                    reused_checkpoint_path(args, regime) if rung == REUSED_RUNG else None
                ),
            )
            achieved = fit["metrics"].get("recall@5")
            ceiling = headroom.get("recall_ceiling@5")
            fit["ceiling_attainment_at_5"] = (
                float(achieved) / float(ceiling) if achieved is not None and ceiling else None
            )
            cell["rungs"][rung] = fit

        divergent = sorted(
            rung for rung, fit in cell["rungs"].items()
            if fit["shared_inputs_sha256"] != shared["sha256"]
        )
        if divergent:  # pragma: no cover - defensive; one store makes this unreachable
            raise ValueError(
                f"{args.dataset}/{regime}: rungs {divergent} did not record the cell's shared "
                "inputs, so this cell is not a controlled comparison"
            )
        distinct = {fit["semantic_rung_fingerprint"]["sha256"] for fit in cell["rungs"].values()}
        if len(distinct) != len(cell["rungs"]):
            raise ValueError(
                f"{args.dataset}/{regime}: two rungs share a semantic fingerprint, so the "
                "comparison has no independent variable"
            )
        cells[regime] = cell

    return {
        "status": STATUS_COMPLETE,
        "stage": "m2b_semantic_minimality",
        "dataset": args.dataset,
        "declaration": "configs/m2b_semantic_minimality.yaml",
        "arm": DECLARED_UNIVERSAL_ARM,
        "runner_arm": RUNNER_UNIVERSAL_ARM,
        "rungs": list(rungs),
        "reused_rung": REUSED_RUNG if REUSED_RUNG in rungs else None,
        "new_rungs": [rung for rung in rungs if rung in NEW_RUNGS],
        "seed": args.seed,
        "queries": len(queries),
        "split": fbc.QUERY_SPLIT,
        "selection": fbc.QUERY_SELECTION,
        "test_split_read": False,
        "holdout_fraction": args.holdout_fraction,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": num_nodes,
        "a64_mainline_family": args.a64_mainline_family if "R3" in regimes else None,
        "device": str(device),
        "provenance": provenance,
        "artifact_root": str(artifact_root),
        "cell_master_root": str(cell_master_root),
        "features_were_loaded_not_rebuilt": (
            "M2B never calls _cell_master_local. Every cell master is M2's, admitted on its "
            "feature build contract rather than on the hash of a declaration file."
        ),
        "cells": cells,
        "systems": {"peak_process_rss_bytes": _m2._peak_rss_bytes()},
    }


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--a64-mainline-family", default=_m2.MAINLINE_FAMILY)
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument("--rungs", nargs="+", default=None, choices=list(ALL_RUNGS))
    parser.add_argument("--seed", type=int, default=DECLARED_SEED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latency-queries", type=int, default=200)
    parser.add_argument("--latency-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--latency-warmup", type=int, default=DEFAULT_WARMUP_QUERIES)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument(
        "--cell-master-root", type=Path, default=None,
        help=(
            "where M2's persisted cell masters live, as <root>/<regime>/cell_features. "
            "Defaults to --artifact-root; kept separate so a fan-out reads M2's sealed "
            "artifacts without writing into them."
        ),
    )
    parser.add_argument(
        "--m2-fits-root", type=Path, default=None,
        help="where M2's completed fits live, for reusing the S3 checkpoint",
    )
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed != DECLARED_SEED:
        parser.error(
            f"M2B declares seed {DECLARED_SEED} only; --seed {args.seed} needs an amendment"
        )
    args.baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "dataset": result["dataset"],
        "cells": {
            regime: {
                rung: {
                    "recall@5": fit["metrics"].get("recall@5"),
                    "total_parameters": fit["parameters"]["total"],
                    "uncached_p95_ms": fit["systems"]["uncached_inference_p95_ms"],
                    "train_seconds": fit["systems"]["train_time_seconds"],
                }
                for rung, fit in cell["rungs"].items()
            }
            for regime, cell in result["cells"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
