"""Paired MLP and message-passing node encoders for retrieval."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


class ResidualMLPEncoder(nn.Module):
    """Graph-blind peer with the same layer count and output interface as a GNN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        *,
        return_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        del edge_index
        h = F.relu(self.input(x))
        intermediates = [h]
        for linear, norm in zip(self.layers, self.norms):
            update = F.dropout(F.relu(linear(h)), p=self.dropout, training=self.training)
            h = norm(h + update)
            intermediates.append(h)
        result = F.normalize(self.output(h), dim=-1)
        return (result, intermediates) if return_layers else result


class MessagePassingEncoder(nn.Module):
    """GCN/SAGE/GATv2/GIN encoder with residuals and exposed layer states."""

    def __init__(
        self,
        kind: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise ImportError("Install the 'graph' extra to construct a GNN encoder") from exc
        factories = {
            "gcn": lambda: GCNConv(hidden_dim, hidden_dim),
            "sage": lambda: SAGEConv(hidden_dim, hidden_dim),
            "gat": lambda: GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False),
            "gin": lambda: GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            ),
        }
        if kind not in factories:
            raise ValueError(f"Unknown GNN kind: {kind}")
        self.kind = kind
        self.input = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(factories[kind]() for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        h = F.relu(self.input(x))
        intermediates = [h]
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, edge_index)
            update = F.dropout(F.relu(update), p=self.dropout, training=self.training)
            h = norm(h + update)
            intermediates.append(h)
        result = F.normalize(self.output(h), dim=-1)
        return (result, intermediates) if return_layers else result


class QueryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(queries), dim=-1)


class DualEncoderRetriever(nn.Module):
    """Dot-product retrieval with a graph-blind or graph-aware node tower."""

    def __init__(self, query_encoder: QueryEncoder, node_encoder: nn.Module):
        super().__init__()
        self.query_encoder = query_encoder
        self.node_encoder = node_encoder

    def forward(
        self,
        query_features: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor | None,
    ) -> torch.Tensor:
        queries = self.query_encoder(query_features)
        nodes = self.node_encoder(node_features, edge_index)
        return queries @ nodes.T


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _closest_mlp_width(
    target_parameters: int,
    input_dim: int,
    output_dim: int,
    num_layers: int,
) -> int:
    candidates = range(8, 2049, 8)
    best_width, best_error = 8, float("inf")
    for width in candidates:
        candidate = ResidualMLPEncoder(input_dim, width, output_dim, num_layers=num_layers)
        error = abs(count_parameters(candidate) - target_parameters)
        if error < best_error:
            best_width, best_error = width, error
    return best_width


def build_parameter_matched_pair(
    kind: str,
    node_input_dim: int,
    query_input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> tuple[DualEncoderRetriever, DualEncoderRetriever, dict[str, float | int]]:
    """Construct a GNN and its closest-width MLP peer with identical query towers."""

    gnn_nodes = MessagePassingEncoder(
        kind,
        node_input_dim,
        hidden_dim,
        output_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    target = count_parameters(gnn_nodes)
    mlp_width = _closest_mlp_width(target, node_input_dim, output_dim, num_layers)
    mlp_nodes = ResidualMLPEncoder(
        node_input_dim,
        mlp_width,
        output_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    query = QueryEncoder(query_input_dim, hidden_dim, output_dim, dropout)
    mlp = DualEncoderRetriever(deepcopy(query), mlp_nodes)
    gnn = DualEncoderRetriever(deepcopy(query), gnn_nodes)
    mlp_count, gnn_count = count_parameters(mlp), count_parameters(gnn)
    return mlp, gnn, {
        "mlp_parameters": mlp_count,
        "gnn_parameters": gnn_count,
        "relative_parameter_gap": abs(mlp_count - gnn_count) / max(gnn_count, 1),
        "mlp_hidden_dim": mlp_width,
    }

