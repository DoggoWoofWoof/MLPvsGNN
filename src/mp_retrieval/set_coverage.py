"""Permutation-invariant set-coverage objective for K-direction Offset models."""

from __future__ import annotations

from itertools import permutations

import torch
import torch.nn.functional as F

from .complete_data import CompleteQuery


_ASSIGNMENT_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def best_injective_assignment_loss(
    local_directional_scores: torch.Tensor,
    positive_local: torch.Tensor,
) -> torch.Tensor:
    """Assign each positive to a distinct direction using the minimum CE cost."""

    if local_directional_scores.ndim != 2:
        raise ValueError("directional scores must have shape [candidates, directions]")
    positive_local = positive_local.to(local_directional_scores.device)
    positive_count = int(positive_local.numel())
    direction_count = int(local_directional_scores.shape[1])
    if positive_count < 1:
        raise ValueError("set assignment requires at least one in-pool positive")
    if positive_count > direction_count:
        raise ValueError(
            f"{positive_count} positives cannot be injectively assigned to {direction_count} directions"
        )
    negative_log_prob = -F.log_softmax(local_directional_scores, dim=0)[positive_local]
    cache_key = (direction_count, positive_count, str(local_directional_scores.device))
    assignments = _ASSIGNMENT_CACHE.get(cache_key)
    if assignments is None:
        assignments = torch.tensor(
            list(permutations(range(direction_count), positive_count)),
            dtype=torch.long,
            device=local_directional_scores.device,
        )
        _ASSIGNMENT_CACHE[cache_key] = assignments
    positive_rows = torch.arange(positive_count, device=local_directional_scores.device)
    costs = negative_log_prob[positive_rows[None, :], assignments].mean(dim=1)
    return costs.min()


def direction_diversity_penalty(
    targets: torch.Tensor,
    *,
    cosine_margin: float,
) -> torch.Tensor:
    """Penalize pairs of normalized relation targets above a cosine margin."""

    if targets.ndim != 3:
        raise ValueError("targets must have shape [queries, directions, hidden]")
    directions = int(targets.shape[1])
    if directions < 2:
        return targets.new_zeros(())
    normalized = F.normalize(targets, dim=-1)
    cosine = normalized @ normalized.transpose(1, 2)
    mask = ~torch.eye(directions, dtype=torch.bool, device=targets.device)
    off_diagonal = cosine[:, mask]
    return F.relu(off_diagonal - cosine_margin).square().mean()


def set_coverage_loss(
    directional_scores: torch.Tensor,
    targets: torch.Tensor,
    batch: list[CompleteQuery],
    lengths: list[int],
    *,
    diversity_weight: float,
    diversity_cosine_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine best injective positive assignment and fixed target diversity."""

    assignment_losses: list[torch.Tensor] = []
    offset = 0
    for query, length in zip(batch, lengths):
        local_scores = directional_scores[offset : offset + length]
        if query.relevant_local.numel():
            assignment_losses.append(
                best_injective_assignment_loss(local_scores, query.relevant_local)
            )
        offset += length
    if not assignment_losses:
        raise RuntimeError("Training batch has no in-pool gold candidates")
    assignment = torch.stack(assignment_losses).mean()
    diversity = direction_diversity_penalty(
        targets,
        cosine_margin=diversity_cosine_margin,
    )
    total = assignment + diversity_weight * diversity
    return total, {"assignment": assignment, "diversity": diversity}
