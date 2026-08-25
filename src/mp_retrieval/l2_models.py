"""Parameter-matched query-local scorers for the Level-2 experiment."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CandidateMLPScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        h = F.gelu(self.input(features))
        for layer, norm in zip(self.layers, self.norms):
            update = F.dropout(F.gelu(layer(h)), p=self.dropout, training=self.training)
            h = norm(h + update)
        return self.output(h).squeeze(-1)


class CandidateGNNScorer(nn.Module):
    def __init__(
        self,
        kind: str,
        input_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the 'graph' extra to run L2 GNN scorers") from exc
        factories = {
            "gcn": lambda: GCNConv(hidden_dim, hidden_dim),
            "sage": lambda: SAGEConv(hidden_dim, hidden_dim),
            "gat": lambda: GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False),
            "gin": lambda: GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            ),
        }
        if kind not in factories:
            raise ValueError(f"Unknown message-passing kind: {kind}")
        self.kind = kind
        self.input = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(factories[kind]() for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.input(features))
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(F.gelu(conv(h, edge_index)), p=self.dropout, training=self.training)
            h = norm(h + update)
        return self.output(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def parameter_matched_scorers(
    kind: str,
    input_dim: int,
    gnn_hidden_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> tuple[CandidateMLPScorer, CandidateGNNScorer, dict[str, float | int]]:
    gnn = CandidateGNNScorer(
        kind,
        input_dim,
        gnn_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    target = count_parameters(gnn)
    widths = range(8, 2049, 8)
    candidates = (
        CandidateMLPScorer(input_dim, width, num_layers=num_layers, dropout=dropout)
        for width in widths
    )
    mlp = min(candidates, key=lambda model: abs(count_parameters(model) - target))
    mlp_count, gnn_count = count_parameters(mlp), count_parameters(gnn)
    return mlp, gnn, {
        "mlp_parameters": mlp_count,
        "gnn_parameters": gnn_count,
        "relative_parameter_gap": abs(mlp_count - gnn_count) / max(gnn_count, 1),
        "mlp_hidden_dim": mlp.input.out_features,
    }
