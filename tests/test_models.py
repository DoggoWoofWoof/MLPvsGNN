import pytest
import torch

from mp_retrieval.l2_models import count_parameters, parameter_matched_scorers
from mp_retrieval.models import ResidualMLPEncoder


def test_mlp_encoder_shape():
    model = ResidualMLPEncoder(7, 16, 8, num_layers=2)
    output, layers = model(torch.randn(5, 7), return_layers=True)
    assert output.shape == (5, 8)
    assert len(layers) == 3
    assert torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_parameter_matched_l2_pair_if_pyg_available():
    pytest.importorskip("torch_geometric")
    mlp, gnn, report = parameter_matched_scorers("gcn", 28, 32, num_layers=2)
    assert report["relative_parameter_gap"] < 0.05
    x = torch.randn(6, 28)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    assert mlp(x, edge_index).shape == (6,)
    assert gnn(x, edge_index).shape == (6,)
    assert count_parameters(mlp) == report["mlp_parameters"]
