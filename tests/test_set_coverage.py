import pytest
import torch

from mp_retrieval.complete_data import CompleteQuery
from mp_retrieval.set_coverage import (
    best_injective_assignment_loss,
    direction_diversity_penalty,
    set_coverage_loss,
)


def _query(positives: list[int]) -> CompleteQuery:
    return CompleteQuery(
        query_index=0,
        query_id="q",
        candidate_index=torch.arange(4),
        relevant_local=torch.tensor(positives),
        relevant_global=torch.tensor(positives),
        anchor_global=0,
        split=0,
    )


def test_assignment_is_invariant_to_positive_and_direction_permutations():
    scores = torch.tensor(
        [
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
        ]
    )
    expected = best_injective_assignment_loss(scores, torch.tensor([0, 1, 2]))
    assert best_injective_assignment_loss(scores, torch.tensor([2, 0, 1])) == pytest.approx(
        float(expected)
    )
    assert best_injective_assignment_loss(scores[:, [2, 0, 3, 1]], torch.tensor([0, 1, 2])) == (
        pytest.approx(float(expected))
    )


def test_assignment_uses_every_positive_and_requires_enough_directions():
    scores = torch.tensor([[8.0, 8.0], [0.0, 0.0], [0.0, 0.0]])
    one_positive = best_injective_assignment_loss(scores, torch.tensor([0]))
    two_positives = best_injective_assignment_loss(scores, torch.tensor([0, 1]))
    assert two_positives > one_positive
    with pytest.raises(ValueError, match="injectively assigned"):
        best_injective_assignment_loss(scores, torch.tensor([0, 1, 2]))


def test_diversity_penalizes_collapsed_directions():
    collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    spread = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    assert direction_diversity_penalty(collapsed, cosine_margin=0.2) > (
        direction_diversity_penalty(spread, cosine_margin=0.2)
    )


def test_set_loss_reports_frozen_components():
    scores = torch.tensor(
        [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0], [0.0, 0.0, 8.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
        requires_grad=True,
    )
    targets = torch.randn(1, 4, 3, requires_grad=True)
    loss, parts = set_coverage_loss(
        scores,
        targets,
        [_query([0, 1, 2, 3])],
        [4],
        diversity_weight=0.1,
        diversity_cosine_margin=0.2,
    )
    assert set(parts) == {"assignment", "diversity"}
    loss.backward()
    assert scores.grad is not None
    assert targets.grad is not None
