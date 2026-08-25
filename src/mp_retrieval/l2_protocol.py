"""Executable fairness contract for paired Level-2 comparisons."""

from __future__ import annotations

import hashlib
import json

import torch

from .l2_data import CandidateQuery


def _update_tensor(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def comparison_contract(
    queries: list[CandidateQuery],
    features: dict[CandidateQuery, torch.Tensor],
    edges: dict[CandidateQuery, torch.Tensor],
    *,
    seeds: list[int],
) -> dict[str, object]:
    """Hash every shared input and state the sole information asymmetry."""

    candidates = hashlib.sha256()
    labels = hashlib.sha256()
    raw_features = hashlib.sha256()
    model_features = hashlib.sha256()
    topology = hashlib.sha256()
    for query in sorted(queries, key=lambda item: item.query_id):
        query_key = query.query_id.encode("utf-8")
        for digest in (candidates, labels, raw_features, model_features, topology):
            digest.update(query_key)
        candidates.update(str(query.split).encode())
        _update_tensor(candidates, query.candidate_index)
        _update_tensor(labels, query.relevant_local)
        _update_tensor(labels, query.relevant_global)
        _update_tensor(raw_features, query.expert_scores)
        _update_tensor(raw_features, query.expert_mask)
        _update_tensor(model_features, features[query])
        _update_tensor(topology, edges[query])
    return {
        "enforced": True,
        "same_candidates": True,
        "same_labels": True,
        "same_listwise_loss": True,
        "same_seed_schedule": True,
        "same_raw_frozen_features": True,
        "same_model_input_features": True,
        "same_optimizer_and_training_loop": True,
        "topology_is_only_extra_gnn_information": True,
        "mlp_topology_argument": None,
        "gnn_topology_argument": "query-induced edge_index",
        "loss": "multi_positive_listwise_cross_entropy",
        "seeds": list(seeds),
        "sha256": {
            "candidates_and_splits": candidates.hexdigest(),
            "labels": labels.hexdigest(),
            "raw_expert_evidence": raw_features.hexdigest(),
            "shared_model_inputs": model_features.hexdigest(),
            "gnn_topology": topology.hexdigest(),
        },
    }
