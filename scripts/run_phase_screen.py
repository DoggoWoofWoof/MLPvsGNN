#!/usr/bin/env python
"""Run validation-only topology/feature phase screening without test access."""

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
from mp_retrieval.local_topology_perturbations import (
    build_or_load_perturbed_topologies,
)
from mp_retrieval.operator_models import model_parameter_counts
from mp_retrieval.protocol import seed_everything
from mp_retrieval.structural_features import (
    StructuralFeatureStore,
    build_or_load_structural_features,
)
from mp_retrieval.topology_store import PackedLocalTopologies
from scripts.run_edge_provenance import _atomic_json, _sha256
from scripts.run_main_table import _state_sha256
from scripts.run_sa_mlp_confirmation import (
    _build_model,
    _fit,
    _score_once,
    validate_candidate_contract,
)

MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
TOPOLOGY_AXES = {"degree_rewire", "random_add", "hub_injection"}
COMPLETE_STATUS = "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE"


def completed_screen_cell(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return an already-registered complete cell, or ``None`` to run it.

    The screen trains one registered seed per cell and has no per-model resume,
    so an unguarded relaunch would retrain and overwrite cells that are already
    complete and integrity-audited.  Reusing a complete record keeps registered
    validation values byte-identical across resumptions; a record that claims
    the same output path under a different frozen contract is an error rather
    than something to overwrite.
    """

    if not args.output.is_file():
        return None
    existing = json.loads(args.output.read_text(encoding="utf-8"))
    if existing.get("status") != COMPLETE_STATUS:
        return None
    contract = existing.get("screen_contract", {})
    intervention = existing.get("intervention", {})
    if (
        existing.get("dataset") != args.dataset
        or existing.get("axis") != args.axis
        or float(existing.get("rate", float("nan"))) != float(args.rate)
        or existing.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
        or int(contract.get("training_seed", -1)) != int(args.training_seed)
        or int(intervention.get("seed", -1)) != int(args.perturbation_seed)
        or contract.get("split_evaluated") != "validation_only"
        or contract.get("test_metrics_computed") is not False
    ):
        raise ValueError("Existing complete phase-screen cell has a different frozen contract")
    if set(existing.get("models", {})) != set(MODEL_NAMES):
        raise ValueError("Existing complete phase-screen cell is missing a registered model")
    if "validation_gnn_minus_qls" not in existing:
        raise ValueError("Existing complete phase-screen cell has no validation contrast")
    return existing


def _mask_node_features(
    nodes: torch.Tensor,
    *,
    rate: float,
    seed: int,
) -> dict[str, Any]:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("Feature-mask rate must be in [0, 1]")
    if rate == 0.0:
        masked = 0
    elif rate == 1.0:
        masked = nodes.numel()
        nodes.zero_()
    else:
        generator = torch.Generator(device=nodes.device).manual_seed(seed)
        mask = torch.rand(
            nodes.shape,
            generator=generator,
            device=nodes.device,
            dtype=torch.float32,
        ) < rate
        masked = int(mask.sum().item())
        nodes.masked_fill_(mask, 0.0)
        del mask
    return {
        "kind": "feature_mask",
        "requested_rate": rate,
        "seed": seed,
        "masked_scalar_entries": masked,
        "total_scalar_entries": nodes.numel(),
        "achieved_rate": masked / max(nodes.numel(), 1),
        "query_embeddings_masked": False,
        "structural_features_changed": False,
    }


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if args.axis not in TOPOLOGY_AXES | {"feature_mask"}:
        raise ValueError(f"Unsupported phase-screen axis: {args.axis}")
    finished = completed_screen_cell(args)
    if finished is not None:
        return finished
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the screen protocol")
    source_contract = validate_candidate_contract(
        args.baseline,
        dataset,
        args.candidate_contract_compatibility,
    )
    if args.baseline["selected_gnn"]["model"] != args.selected_gnn:
        raise ValueError("Selected GNN differs from the frozen confirmation")
    train_queries = dataset.split(QuerySplit.TRAIN)
    validation_queries = dataset.split(QuerySplit.VALIDATION)
    if not train_queries or not validation_queries:
        raise RuntimeError("Phase screening requires canonical train/validation splits")
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
    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    result = {
        "status": "PHASE_SCREEN_IN_PROGRESS",
        "dataset": args.dataset,
        "axis": args.axis,
        "rate": args.rate,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": source_contract,
        "intervention": intervention,
        "screen_contract": {
            "split_evaluated": "validation_only",
            "test_metrics_computed": False,
            "training_seed": args.training_seed,
            "same_intervention_for_both_models": True,
            "same_embeddings_candidates_labels_loss_optimizer_epochs": True,
            "same_selected_gnn_family": True,
            "topology_axes_change_query_local_topology_for_both_models": True,
            "topology_axes_keep_corpus_static_features_clean": True,
            "feature_mask_changes_raw_node_features_for_both_models": True,
        },
        "models": {},
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in {"baseline", "feature_config"}
        },
    }
    for model_name in MODEL_NAMES:
        seed_everything(args.training_seed)
        model = _build_model(
            model_name,
            dataset,
            features,
            args.selected_gnn,
            target_parameters,
            args,
        )
        counts = model_parameter_counts(model)
        if model_name == "seed_aware_gnn":
            if int(counts["parameters"]) - target_parameters != args.hidden_dim:
                raise ValueError("Seed-aware GNN parameter count changed")
        elif abs(int(counts["parameters"]) - target_parameters) > args.max_parameter_difference:
            raise ValueError("QLS-MLP parameter match exceeds the frozen tolerance")
        model, training = _fit(
            model_name,
            model,
            train_queries,
            validation_queries,
            nodes,
            query_embeddings,
            topologies,
            features,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.training_seed,
        )
        metrics, _rows, telemetry = _score_once(
            model_name,
            model,
            validation_queries,
            nodes,
            query_embeddings,
            topologies,
            features,
            device,
            batch_size=args.batch_size,
            ks=tuple(args.ks),
            timed=True,
        )
        checkpoint_path = args.checkpoint_root / model_name / "seed_0.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        result["models"][model_name] = {
            "parameters": counts,
            "validation_metrics": metrics,
            "training": training,
            "validation_telemetry": telemetry,
            "checkpoint_sha256": _state_sha256(model.state_dict()),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_file_sha256": _sha256(checkpoint_path),
        }
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["validation_gnn_minus_qls"] = {
        metric: float(result["models"]["seed_aware_gnn"]["validation_metrics"][metric])
        - float(result["models"]["sa_mlp"]["validation_metrics"][metric])
        for metric in result["models"]["sa_mlp"]["validation_metrics"]
    }
    result["status"] = "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE"
    _atomic_json(args.output, result)
    if checkpoint_hook is not None:
        checkpoint_hook()
    return result


if __name__ == "__main__":
    raise SystemExit("Use scripts/modal_phase_screen.py for the frozen execution")
