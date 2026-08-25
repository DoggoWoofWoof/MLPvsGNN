"""Matched query-local scorers for the Level-2 experiment."""

from __future__ import annotations

from statistics import median
import time
from typing import Iterable

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


def build_mlp_scorer(
    input_dim: int,
    hidden_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> CandidateMLPScorer:
    return CandidateMLPScorer(
        input_dim,
        hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_gnn_scorer(
    kind: str,
    input_dim: int,
    hidden_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> CandidateGNNScorer:
    return CandidateGNNScorer(
        kind,
        input_dim,
        hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def parameter_matched_width(
    kind: str,
    input_dim: int,
    gnn_hidden_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.2,
    widths: Iterable[int] = range(8, 2049, 8),
) -> tuple[int, dict[str, float | int | str]]:
    """Choose the MLP width with the closest trainable parameter count."""

    gnn = build_gnn_scorer(
        kind,
        input_dim,
        gnn_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    target = count_parameters(gnn)
    candidates = [
        build_mlp_scorer(input_dim, width, num_layers=num_layers, dropout=dropout)
        for width in widths
    ]
    if not candidates:
        raise ValueError("At least one candidate MLP width is required")
    mlp = min(candidates, key=lambda model: abs(count_parameters(model) - target))
    mlp_count, gnn_count = count_parameters(mlp), count_parameters(gnn)
    return mlp.input.out_features, {
        "method": "trainable_parameter_count",
        "mlp_parameters": mlp_count,
        "gnn_parameters": gnn_count,
        "relative_parameter_gap": abs(mlp_count - gnn_count) / max(gnn_count, 1),
        "mlp_hidden_dim": mlp.input.out_features,
        "gnn_hidden_dim": gnn_hidden_dim,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _training_step_seconds(
    model: nn.Module,
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    uses_topology: bool,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> float:
    """Median forward/backward time over fixed calibration queries.

    This is a hardware-specific compute proxy. It deliberately excludes data
    loading and optimizer updates, and never uses validation or test outcomes.
    """

    model = model.to(device)
    model.train()
    timings: list[float] = []
    total_rounds = warmups + repeats
    for round_idx in range(total_rounds):
        started = time.perf_counter()
        for features, edge_index, relevant_local in examples:
            x = features.to(device)
            edges = edge_index.to(device) if uses_topology else None
            positives = relevant_local.to(device)
            scores = model(x, edges)
            target = torch.zeros_like(scores)
            if positives.numel():
                target[positives] = 1.0 / positives.numel()
                loss = -(F.log_softmax(scores, dim=0) * target).sum()
            else:
                loss = scores.square().mean()
            model.zero_grad(set_to_none=True)
            loss.backward()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if round_idx >= warmups:
            timings.append(elapsed)
    return float(median(timings))


def compute_matched_width(
    kind: str,
    input_dim: int,
    gnn_hidden_dim: int,
    calibration_examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    num_layers: int = 2,
    dropout: float = 0.2,
    widths: Iterable[int] = range(4, 513),
    warmups: int = 1,
    repeats: int = 3,
) -> tuple[int, dict[str, float | int | str | list[dict[str, float | int]]]]:
    """Choose an MLP width matched on a sparse multiply-accumulate proxy.

    The matching target uses each calibration graph's actual node and edge
    count. It is deterministic and outcome-free. Wall time is retained only as
    a hardware diagnostic because short CPU/GPU timings are not monotonic or
    reproducible enough to define the scientific comparison arm.
    """

    if not calibration_examples:
        raise ValueError("Compute matching requires at least one training query")
    multipliers = {"gcn": 1, "sage": 2, "gat": 2, "gin": 2}
    hidden_transforms = multipliers[kind]
    sizes = [(int(x.shape[0]), int(edges.shape[1])) for x, edges, _ in calibration_examples]

    def mlp_macs(width: int) -> int:
        return sum(
            nodes * input_dim * width
            + num_layers * nodes * width * width
            + nodes * width
            for nodes, _edges in sizes
        )

    def gnn_macs() -> int:
        return sum(
            nodes * input_dim * gnn_hidden_dim
            + num_layers
            * (
                hidden_transforms * nodes * gnn_hidden_dim * gnn_hidden_dim
                + edge_count * gnn_hidden_dim
            )
            + nodes * gnn_hidden_dim
            for nodes, edge_count in sizes
        )

    target_macs = gnn_macs()
    measurements: list[dict[str, float | int]] = []
    for width in widths:
        macs = mlp_macs(int(width))
        measurements.append(
            {
                "hidden_dim": int(width),
                "macs": macs,
                "relative_gap": abs(macs - target_macs) / max(target_macs, 1),
            }
        )
    best = min(measurements, key=lambda row: float(row["relative_gap"]))
    selected_width = int(best["hidden_dim"])
    gnn = build_gnn_scorer(
        kind,
        input_dim,
        gnn_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    mlp = build_mlp_scorer(
        input_dim,
        selected_width,
        num_layers=num_layers,
        dropout=dropout,
    )
    gnn_seconds = _training_step_seconds(
        gnn,
        calibration_examples,
        uses_topology=True,
        device=device,
        warmups=warmups,
        repeats=repeats,
    )
    mlp_seconds = _training_step_seconds(
        mlp,
        calibration_examples,
        uses_topology=False,
        device=device,
        warmups=warmups,
        repeats=repeats,
    )
    return int(best["hidden_dim"]), {
        "method": "analytical_sparse_multiply_accumulate_proxy",
        "gnn_operator_hidden_transform_multiplier": hidden_transforms,
        "calibration_queries": len(calibration_examples),
        "calibration_sizes": [
            {"nodes": nodes, "edges": edges} for nodes, edges in sizes
        ],
        "gnn_macs": target_macs,
        "mlp_macs": int(best["macs"]),
        "relative_compute_gap": float(best["relative_gap"]),
        "mlp_hidden_dim": selected_width,
        "gnn_hidden_dim": gnn_hidden_dim,
        "measurements": measurements,
        "wall_clock_is_diagnostic_not_matching_criterion": True,
        "wall_clock": {
            "device": str(device),
            "warmups": warmups,
            "repeats": repeats,
            "gnn_seconds": gnn_seconds,
            "mlp_seconds": mlp_seconds,
            "relative_gap": abs(mlp_seconds - gnn_seconds) / max(gnn_seconds, 1e-12),
        },
    }


def parameter_matched_scorers(
    kind: str,
    input_dim: int,
    gnn_hidden_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> tuple[CandidateMLPScorer, CandidateGNNScorer, dict[str, float | int]]:
    width, report = parameter_matched_width(
        kind,
        input_dim,
        gnn_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return (
        build_mlp_scorer(input_dim, width, num_layers=num_layers, dropout=dropout),
        build_gnn_scorer(
            kind,
            input_dim,
            gnn_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ),
        report,
    )
