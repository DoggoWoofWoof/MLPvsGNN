"""Shared candidate features for a fair L2 MLP/GNN comparison."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .l2_data import CandidateQuery


def present_only_stats(queries: list[CandidateQuery]) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only mean/std that excludes unavailable expert entries."""

    if not queries:
        raise ValueError("At least one training query is required")
    num_experts = queries[0].expert_scores.shape[1]
    total = torch.zeros(num_experts, dtype=torch.float64)
    total_sq = torch.zeros(num_experts, dtype=torch.float64)
    count = torch.zeros(num_experts, dtype=torch.float64)
    for query in queries:
        scores = query.expert_scores.double()
        mask = query.expert_mask.double()
        total += (scores * mask).sum(dim=0)
        total_sq += (scores.square() * mask).sum(dim=0)
        count += mask.sum(dim=0)
    safe = count.clamp_min(1.0)
    mean = total / safe
    variance = (total_sq / safe - mean.square()).clamp_min(1e-12)
    return mean.float(), variance.sqrt().float()


def _percentile_and_topk(scores: torch.Tensor, mask: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    count, experts = scores.shape
    percentile = torch.zeros_like(scores)
    topk = torch.zeros_like(scores)
    for expert in range(experts):
        valid = torch.nonzero(mask[:, expert], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        values = scores[valid, expert]
        order = torch.argsort(values)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(order.numel())
        percentile[valid, expert] = ranks.float() / max(order.numel() - 1, 1)
        best = valid[torch.topk(values, min(k, values.numel())).indices]
        topk[best, expert] = 1.0
    return percentile, topk


def _query_state(z: torch.Tensor, mask: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
    """Scale-free confidence and agreement signals, repeated for every candidate."""

    count, experts = z.shape
    margins = torch.zeros(experts)
    entropies = torch.zeros(experts)
    availability = mask.any(dim=0).float()
    agreement = torch.zeros(experts)
    for expert in range(experts):
        valid = mask[:, expert]
        values = z[valid, expert]
        if values.numel():
            top = torch.topk(values, min(2, values.numel())).values
            margins[expert] = top[0] - top[-1] if top.numel() > 1 else 0.0
            probabilities = F.softmax(values, dim=0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            entropies[expert] = entropy / max(math.log(values.numel()), 1.0)
        pairwise = []
        this = topk[:, expert] > 0
        for other in range(experts):
            if other == expert:
                continue
            that = topk[:, other] > 0
            denominator = max(min(int(this.sum()), int(that.sum())), 1)
            pairwise.append(float((this & that).sum()) / denominator)
        agreement[expert] = sum(pairwise) / max(len(pairwise), 1)
    return torch.cat([margins, entropies, agreement, availability])


def build_candidate_features(
    query: CandidateQuery,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    topk: int = 5,
) -> torch.Tensor:
    """Build the exact same candidate inputs for the MLP and every GNN."""

    mask = query.expert_mask
    z = (query.expert_scores - mean) / std.clamp_min(1e-6)
    z = torch.where(mask, z, torch.zeros_like(z))
    percentile, member = _percentile_and_topk(query.expert_scores, mask, topk)
    state = _query_state(z, mask, member)
    repeated_state = state.unsqueeze(0).expand(z.shape[0], -1)
    return torch.cat([z, mask.float(), percentile, member, repeated_state], dim=1)

