import torch

from mp_retrieval.graph_stats import degree_statistics, feature_edge_alignment
from mp_retrieval.representation import effective_rank, normalized_dirichlet_energy
from mp_retrieval.retrieval import metrics_for_ranking, paired_bootstrap_delta


def test_multi_positive_retrieval_metrics():
    metrics = metrics_for_ranking([4, 2, 0, 1, 3], [2, 3], ks=(1, 2, 5))
    assert metrics["hit@1"] == 0.0
    assert metrics["recall@2"] == 0.5
    assert metrics["full_coverage@5"] == 1.0
    assert metrics["mrr"] == 0.5


def test_paired_bootstrap_direction():
    result = paired_bootstrap_delta([1, 1, 1, 1], [0, 0, 0, 0], samples=500, seed=2)
    assert result["delta"] == 1.0
    assert result["ci_low"] == 1.0


def test_graph_and_representation_statistics_are_finite():
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    features = torch.eye(4)
    degree = degree_statistics(edges, 4)
    alignment = feature_edge_alignment(features, edges, max_edges=10)
    assert degree["degree_mean"] == 2.0
    assert alignment["edge_cosine_mean"] == 0.0
    assert effective_rank(features) > 2.9
    assert normalized_dirichlet_energy(features, edges) > 0

