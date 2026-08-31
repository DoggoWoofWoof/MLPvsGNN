#!/usr/bin/env python
"""Run one frozen five-seed phase-crossover confirmation cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.local_topology_perturbations import build_or_load_perturbed_topologies
from mp_retrieval.operator_models import model_parameter_counts
from mp_retrieval.protocol import seed_everything
from mp_retrieval.structural_features import (
    StructuralFeatureStore,
    build_or_load_structural_features,
)
from mp_retrieval.topology_store import PackedLocalTopologies
from scripts.run_confirmation import _aggregate_runs
from scripts.run_edge_provenance import _atomic_json, _sha256
from scripts.run_main_table import _hop_aggregates, _state_sha256
from scripts.run_phase_screen import TOPOLOGY_AXES, _mask_node_features
from scripts.run_sa_mlp_confirmation import (
    PACKED_QUERY_METRICS,
    _build_model,
    _fit,
    _query_order_sha256,
    _repeated_score,
    _rows_to_array,
    validate_candidate_contract,
)

MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")


def _screen_seed_zero(
    path: Path,
    *,
    dataset: str,
    axis: str,
    rate: float,
    fingerprint: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE"
        or payload.get("dataset") != dataset
        or payload.get("axis") != axis
        or float(payload.get("rate", -1)) != rate
        or payload.get("data_fingerprint_sha256") != fingerprint
        or payload["screen_contract"].get("test_metrics_computed") is not False
        or int(payload["screen_contract"].get("training_seed", -1)) != 0
    ):
        raise ValueError("Validation-screen seed-0 reuse contract failed")
    return payload


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if args.rate <= 0.0:
        raise ValueError("Clean rate zero must reuse the sealed confirmation, not be rerun")
    if args.axis not in TOPOLOGY_AXES | {"feature_mask"}:
        raise ValueError(f"Unsupported phase-confirmation axis: {args.axis}")
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the frozen protocol")
    source_contract = validate_candidate_contract(
        args.baseline,
        dataset,
        args.candidate_contract_compatibility,
    )
    if args.baseline["selected_gnn"]["model"] != args.selected_gnn:
        raise ValueError("Selected GNN differs from the frozen confirmation")
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    if any(not rows for rows in splits.values()):
        raise RuntimeError("Phase confirmation requires all canonical splits")
    clean_topologies = PackedLocalTopologies.load(args.clean_topology_cache)
    if args.axis in TOPOLOGY_AXES:
        topologies, intervention = build_or_load_perturbed_topologies(
            clean_topologies,
            dataset.queries,
            args.topology_cache,
            kind=args.axis,
            rate=args.rate,
            seed=args.perturbation_seed,
        )
        feature_fingerprint = hashlib.sha256(
            (
                args.data_fingerprint_sha256
                + intervention["contract_sha256"]
                + "clean_global_static_features"
            ).encode("utf-8")
        ).hexdigest()
        features = build_or_load_structural_features(
            dataset,
            topologies,
            args.feature_cache,
            source_fingerprint=feature_fingerprint,
            config=args.feature_config,
        )
    else:
        topologies = clean_topologies
        features = StructuralFeatureStore.load(args.clean_feature_cache)
        intervention = None
    if checkpoint_hook is not None:
        checkpoint_hook()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        nodes = torch.from_numpy(np.asarray(dataset.node_array)).to(
            device=device, dtype=torch.float32
        )
        query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
            device=device, dtype=torch.float32
        )
    if args.axis == "feature_mask":
        intervention = _mask_node_features(
            nodes,
            rate=args.rate,
            seed=args.perturbation_seed,
        )
    assert intervention is not None
    screen = _screen_seed_zero(
        args.screen_result,
        dataset=args.dataset,
        axis=args.axis,
        rate=args.rate,
        fingerprint=args.data_fingerprint_sha256,
    )
    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    if args.output.is_file():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            result.get("dataset") != args.dataset
            or result.get("axis") != args.axis
            or float(result.get("rate", -1)) != args.rate
            or result.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
        ):
            raise ValueError("Existing phase-confirmation result has a different contract")
    else:
        result = {
            "status": "PHASE_CONFIRMATION_IN_PROGRESS",
            "dataset": args.dataset,
            "axis": args.axis,
            "rate": args.rate,
            "data_fingerprint_sha256": args.data_fingerprint_sha256,
            "candidate_contract": source_contract,
            "intervention": intervention,
            "data": {
                "queries": len(dataset.queries),
                "splits": {split.name.lower(): len(rows) for split, rows in splits.items()},
                "test_query_order_sha256": _query_order_sha256(splits[QuerySplit.TEST]),
            },
            "confirmation_contract": {
                "selected_by_locked_validation_only_rule": True,
                "same_intervention_for_both_models": True,
                "same_embeddings_candidates_labels_loss_optimizer_epochs": True,
                "same_selected_gnn_family": True,
                "seed_zero_validation_checkpoint_reused_without_test_peeking": True,
                "test_selected_rate": False,
                "test_evaluations_per_model_seed": 1,
            },
            "models": {model: {"seeds": {}} for model in MODEL_NAMES},
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in {"baseline", "feature_config"}
            },
        }
    query_arrays: dict[str, np.ndarray] = {}
    if args.query_metrics_output.is_file():
        with np.load(args.query_metrics_output) as packed:
            query_arrays = {
                key: np.asarray(packed[key])
                for key in packed.files
                if key not in {"metric_names", "query_order_sha256"}
            }

    def checkpoint() -> None:
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    for model_name in MODEL_NAMES:
        for seed in args.seeds:
            seed_key = str(seed)
            if seed_key in result["models"][model_name]["seeds"]:
                continue
            seed_everything(seed)
            model = _build_model(
                model_name,
                dataset,
                features,
                args.selected_gnn,
                target_parameters,
                args,
            ).to(device)
            counts = model_parameter_counts(model)
            if model_name == "seed_aware_gnn":
                if int(counts["parameters"]) - target_parameters != args.hidden_dim:
                    raise ValueError("Seed-aware GNN parameter count changed")
            elif (
                abs(int(counts["parameters"]) - target_parameters)
                > args.max_parameter_difference
            ):
                raise ValueError("QLS-MLP parameter match exceeds the frozen tolerance")
            if seed == 0:
                source = screen["models"][model_name]
                checkpoint_path = Path(source["checkpoint_path"])
                if _sha256(checkpoint_path) != source["checkpoint_file_sha256"]:
                    raise ValueError("Validation-screen seed-0 checkpoint failed SHA-256")
                state = torch.load(checkpoint_path, map_location=device, weights_only=True)
                if _state_sha256(state) != source["checkpoint_sha256"]:
                    raise ValueError("Validation-screen seed-0 state failed SHA-256")
                model.load_state_dict(state)
                training = source["training"]
                checkpoint_origin = "validation_screen_seed_0_reused"
            else:
                model, training = _fit(
                    model_name,
                    model,
                    splits[QuerySplit.TRAIN],
                    splits[QuerySplit.VALIDATION],
                    nodes,
                    query_embeddings,
                    topologies,
                    features,
                    device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=seed,
                )
                checkpoint_origin = "phase_confirmation_training"
            metrics, rows, inference = _repeated_score(
                model_name,
                model,
                splits[QuerySplit.TEST],
                nodes,
                query_embeddings,
                topologies,
                features,
                device,
                batch_size=args.batch_size,
                ks=tuple(args.ks),
                repeats=args.inference_repeats,
            )
            output_checkpoint = args.checkpoint_root / model_name / f"seed_{seed}.pt"
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_checkpoint)
            result["models"][model_name]["parameters"] = counts
            result["models"][model_name]["seeds"][seed_key] = {
                "metrics": metrics,
                "training": training,
                "inference": inference,
                "by_hop": _hop_aggregates(splits[QuerySplit.TEST], rows),
                "checkpoint_origin": checkpoint_origin,
                "checkpoint_sha256": _state_sha256(model.state_dict()),
                "checkpoint_path": str(output_checkpoint),
                "checkpoint_file_sha256": _sha256(output_checkpoint),
            }
            query_arrays[f"{model_name}_seed_{seed}"] = _rows_to_array(rows)
            args.query_metrics_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.query_metrics_output,
                metric_names=np.asarray(PACKED_QUERY_METRICS),
                query_order_sha256=np.asarray(result["data"]["test_query_order_sha256"]),
                **query_arrays,
            )
            checkpoint()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for model_name in MODEL_NAMES:
        runs = result["models"][model_name]["seeds"]
        if sorted(map(int, runs)) != sorted(args.seeds):
            raise RuntimeError(f"Incomplete phase-confirmation seed set: {model_name}")
        result["models"][model_name]["aggregate"] = _aggregate_runs(runs)
    result["query_metrics"] = {
        "path": str(args.query_metrics_output),
        "sha256": _sha256(args.query_metrics_output),
        "format": "npz_float32_test_query_order_v1",
        "metrics": list(PACKED_QUERY_METRICS),
        "arrays": sorted(query_arrays),
    }
    result["status"] = "PHASE_CONFIRMATION_CELL_COMPLETE"
    checkpoint()
    return result


if __name__ == "__main__":
    raise SystemExit("Use the frozen Modal phase-confirmation launcher")
