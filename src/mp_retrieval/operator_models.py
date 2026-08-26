"""Clean operator-vs-message-passing models for the complete-data screen."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SCREEN_MODELS = (
    "plain_mlp",
    "offset_mlp",
    "offset_mlp_k4",
    "gcn",
    "sage",
    "gat",
    "gin",
)
COVERAGE_OFFSET_MODEL = "offset_mlp_set_k4"
STRUCTURE_AWARE_MODELS = (
    "interaction",
    "static_structure",
    "query_local_structure",
    "sa_mlp",
)
SEED_ONLY_MODEL = "seed_only"
EXPLICIT_FEATURE_MODELS = (*STRUCTURE_AWARE_MODELS, SEED_ONLY_MODEL)


class OperatorModel(nn.Module):
    uses_topology = False

    def __init__(self, embedding_dim: int, hidden_dim: int, *, dropout: float, temperature: float):
        super().__init__()
        self.node_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.query_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.dropout = float(dropout)
        self.temperature = float(temperature)

    def project_nodes(self, nodes: torch.Tensor) -> torch.Tensor:
        return F.normalize(F.gelu(self.node_projection(nodes)), dim=-1)

    def project_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.query_projection(queries))

    def similarities(
        self,
        nodes: torch.Tensor,
        targets: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        if targets.ndim == 2:
            score = (nodes * targets[batch_index]).sum(dim=-1)
        else:
            score = (nodes[:, None, :] * targets[batch_index]).sum(dim=-1).max(dim=-1).values
        return score / self.temperature


class PlainMLP(OperatorModel):
    """Non-graph target predictor using only the frozen query embedding."""

    def __init__(self, embedding_dim: int, hidden_dim: int, *, dropout: float, temperature: float):
        super().__init__(embedding_dim, hidden_dim, dropout=dropout, temperature=temperature)
        self.target = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        queries: torch.Tensor,
        anchors: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del anchors, edge_index
        node_state = self.project_nodes(nodes)
        query_state = self.project_queries(queries)
        target = F.normalize(query_state + self.target(query_state), dim=-1)
        return self.similarities(node_state, target, batch_index)


class OffsetMLP(OperatorModel):
    """Predict one or more query-conditioned directions from a dense anchor."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        *,
        directions: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__(embedding_dim, hidden_dim, dropout=dropout, temperature=temperature)
        if directions < 1:
            raise ValueError("directions must be positive")
        self.directions = int(directions)
        self.relation = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, directions * hidden_dim),
        )

    def relational_targets(
        self,
        queries: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        query_state = self.project_queries(queries)
        anchor_state = self.project_nodes(anchors)
        offsets = self.relation(torch.cat([query_state, anchor_state], dim=-1)).view(
            queries.shape[0], self.directions, -1
        )
        return F.normalize(anchor_state[:, None, :] + offsets, dim=-1)

    def directional_scores(
        self,
        nodes: torch.Tensor,
        queries: torch.Tensor,
        anchors: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del edge_index
        node_state = self.project_nodes(nodes)
        targets = self.relational_targets(queries, anchors)
        scores = (node_state[:, None, :] * targets[batch_index]).sum(dim=-1)
        return scores / self.temperature, targets

    def forward(
        self,
        nodes: torch.Tensor,
        queries: torch.Tensor,
        anchors: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores, _targets = self.directional_scores(
            nodes,
            queries,
            anchors,
            batch_index,
            edge_index,
        )
        return scores.max(dim=-1).values


class SetCoverageOffsetMLP(OffsetMLP):
    """K-direction Offset model trained with the preregistered set objective."""

    uses_set_coverage_objective = True


class MessagePassingOperator(OperatorModel):
    """Query-scored candidate encoder whose only privileged input is topology."""

    uses_topology = True

    def __init__(
        self,
        kind: str,
        embedding_dim: int,
        hidden_dim: int,
        *,
        layers: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__(embedding_dim, hidden_dim, dropout=dropout, temperature=temperature)
        try:
            from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the graph extra to run message-passing operators") from exc
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
            raise ValueError(f"Unknown message-passing operator: {kind}")
        self.kind = kind
        self.query_target = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList(factories[kind]() for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))

    def forward(
        self,
        nodes: torch.Tensor,
        queries: torch.Tensor,
        anchors: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del anchors
        if edge_index is None:
            raise ValueError("A message-passing operator requires edge_index")
        node_state = self.project_nodes(nodes)
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(
                F.gelu(conv(node_state, edge_index)),
                p=self.dropout,
                training=self.training,
            )
            node_state = norm(node_state + update)
        node_state = F.normalize(node_state, dim=-1)
        query_state = self.project_queries(queries)
        target = F.normalize(query_state + self.query_target(query_state), dim=-1)
        return self.similarities(node_state, target, batch_index)


class SeedAwareMessagePassingOperator(MessagePassingOperator):
    """Frozen GNN architecture with one additional binary retrieval-seed input."""

    uses_retrieval_seed = True

    def __init__(
        self,
        kind: str,
        embedding_dim: int,
        hidden_dim: int,
        *,
        layers: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__(
            kind,
            embedding_dim,
            hidden_dim,
            layers=layers,
            dropout=dropout,
            temperature=temperature,
        )
        # Concatenating a scalar before the otherwise-identical projection adds
        # exactly ``hidden_dim`` trainable parameters to the frozen GNN family.
        self.node_projection = nn.Linear(embedding_dim + 1, hidden_dim, bias=False)

    def forward_seed_aware(
        self,
        nodes: torch.Tensor,
        queries: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor,
        seed_indicator: torch.Tensor,
    ) -> torch.Tensor:
        if seed_indicator.shape != (nodes.shape[0], 1):
            raise ValueError(
                f"Expected seed indicators {(nodes.shape[0], 1)}, got {tuple(seed_indicator.shape)}"
            )
        node_state = F.normalize(
            F.gelu(self.node_projection(torch.cat([nodes, seed_indicator], dim=-1))),
            dim=-1,
        )
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(
                F.gelu(conv(node_state, edge_index)),
                p=self.dropout,
                training=self.training,
            )
            node_state = norm(node_state + update)
        node_state = F.normalize(node_state, dim=-1)
        query_state = self.project_queries(queries)
        target = F.normalize(query_state + self.query_target(query_state), dim=-1)
        return self.similarities(node_state, target, batch_index)


def explicit_feature_input_dim(
    name: str,
    projection_dim: int,
    static_dim: int,
    local_dim: int,
) -> int:
    if name not in EXPLICIT_FEATURE_MODELS:
        raise ValueError(f"Unknown explicit-feature model: {name}")
    dimension = 2 * projection_dim
    if name in {"interaction", "sa_mlp", SEED_ONLY_MODEL}:
        dimension += 2 * projection_dim + 2
    if name in {"static_structure", "sa_mlp"}:
        dimension += static_dim
    if name in {"query_local_structure", "sa_mlp", SEED_ONLY_MODEL}:
        dimension += local_dim
    return dimension


def parameter_matched_head_width(
    *,
    embedding_dim: int,
    projection_dim: int,
    input_dim: int,
    target_parameters: int,
) -> int:
    """Choose the deterministic scalar-head width nearest a frozen target."""

    projection_parameters = 2 * embedding_dim * projection_dim

    def total(width: int) -> int:
        return projection_parameters + input_dim * width + width + width + 1

    candidates = range(1, 513)
    return min(candidates, key=lambda width: (abs(total(width) - target_parameters), width))


class ExplicitFeatureMLP(OperatorModel):
    """Candidate scorer over compact fixed descriptors, with no adjacency input."""

    uses_topology = False
    uses_structural_features = True

    def __init__(
        self,
        name: str,
        embedding_dim: int,
        projection_dim: int,
        *,
        static_dim: int,
        local_dim: int,
        head_dim: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__(
            embedding_dim,
            projection_dim,
            dropout=dropout,
            temperature=temperature,
        )
        if name not in EXPLICIT_FEATURE_MODELS:
            raise ValueError(f"Unknown explicit-feature model: {name}")
        self.kind = name
        self.include_interactions = name in {"interaction", "sa_mlp", SEED_ONLY_MODEL}
        self.include_static = name in {"static_structure", "sa_mlp"}
        self.include_local = name in {"query_local_structure", "sa_mlp", SEED_ONLY_MODEL}
        self.static_dim = static_dim if self.include_static else 0
        self.local_dim = local_dim if self.include_local else 0
        input_dim = explicit_feature_input_dim(
            name,
            projection_dim,
            static_dim,
            local_dim,
        )
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    @property
    def structural_dim(self) -> int:
        return self.static_dim + self.local_dim

    def forward_explicit(
        self,
        nodes: torch.Tensor,
        queries: torch.Tensor,
        batch_index: torch.Tensor,
        structural_features: torch.Tensor | None,
    ) -> torch.Tensor:
        raw_nodes = F.gelu(self.node_projection(nodes))
        raw_queries = F.gelu(self.query_projection(queries))[batch_index]
        node_state = F.normalize(raw_nodes, dim=-1)
        query_state = F.normalize(raw_queries, dim=-1)
        pieces = [query_state, node_state]
        if self.include_interactions:
            pieces.extend(
                [
                    query_state * node_state,
                    torch.abs(query_state - node_state),
                    (query_state * node_state).sum(dim=-1, keepdim=True),
                    (raw_queries * raw_nodes).sum(dim=-1, keepdim=True)
                    / raw_nodes.shape[-1] ** 0.5,
                ]
            )
        if self.structural_dim:
            if structural_features is None:
                raise ValueError(f"{self.kind} requires fixed structural features")
            if structural_features.shape != (nodes.shape[0], self.structural_dim):
                raise ValueError(
                    f"Expected structural features {(nodes.shape[0], self.structural_dim)}, "
                    f"got {tuple(structural_features.shape)}"
                )
            pieces.append(structural_features)
        elif structural_features is not None and structural_features.shape[1] != 0:
            raise ValueError(f"{self.kind} does not accept structural features")
        return self.scorer(torch.cat(pieces, dim=-1)).squeeze(-1) / self.temperature


def build_explicit_feature_mlp(
    name: str,
    embedding_dim: int,
    projection_dim: int,
    *,
    static_dim: int,
    local_dim: int,
    target_parameters: int,
    dropout: float,
    temperature: float,
) -> ExplicitFeatureMLP:
    input_dim = explicit_feature_input_dim(
        name,
        projection_dim,
        static_dim,
        local_dim,
    )
    head_dim = parameter_matched_head_width(
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        input_dim=input_dim,
        target_parameters=target_parameters,
    )
    return ExplicitFeatureMLP(
        name,
        embedding_dim,
        projection_dim,
        static_dim=static_dim,
        local_dim=local_dim,
        head_dim=head_dim,
        dropout=dropout,
        temperature=temperature,
    )


def build_seed_aware_message_passing(
    kind: str,
    embedding_dim: int,
    hidden_dim: int,
    *,
    layers: int = 1,
    dropout: float = 0.2,
    temperature: float = 0.07,
) -> SeedAwareMessagePassingOperator:
    return SeedAwareMessagePassingOperator(
        kind,
        embedding_dim,
        hidden_dim,
        layers=layers,
        dropout=dropout,
        temperature=temperature,
    )


def build_operator_model(
    name: str,
    embedding_dim: int,
    hidden_dim: int,
    *,
    layers: int = 1,
    offset_directions: int = 4,
    dropout: float = 0.2,
    temperature: float = 0.07,
) -> OperatorModel:
    if name == "plain_mlp":
        return PlainMLP(
            embedding_dim,
            hidden_dim,
            dropout=dropout,
            temperature=temperature,
        )
    if name == COVERAGE_OFFSET_MODEL:
        return SetCoverageOffsetMLP(
            embedding_dim,
            hidden_dim,
            directions=offset_directions,
            dropout=dropout,
            temperature=temperature,
        )
    if name in {"offset_mlp", "offset_mlp_k4"}:
        return OffsetMLP(
            embedding_dim,
            hidden_dim,
            directions=1 if name == "offset_mlp" else offset_directions,
            dropout=dropout,
            temperature=temperature,
        )
    if name in {"gcn", "sage", "gat", "gin"}:
        return MessagePassingOperator(
            name,
            embedding_dim,
            hidden_dim,
            layers=layers,
            dropout=dropout,
            temperature=temperature,
        )
    raise ValueError(f"Unknown operator model: {name}")


def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
    }
