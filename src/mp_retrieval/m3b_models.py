"""The three M3B arms (configs/m3b_controlled_comparison.yaml#arms).

One shared input block turns a packed batch -- the compiled scalars, the served
embeddings of the pool nodes and of the query, the pool-graph edges and the
seed-reach weights -- into one hidden vector per candidate; one shared residual
readout adds a zero-initialised correction to the within-query z-score of the
fixed base column. QLS-U puts an MLP between the two; the universal GAT puts
GATv2 layers over the pool-graph edges; GAT-NO-MP is the same GAT module run
with no message edges, so every layer sees its own node through the self-loop
and nothing else. At step 0 every arm returns the fixed base score exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from mp_retrieval.m2b_semantic_control import ProjectionSemanticHead
from mp_retrieval.m2d_stage1_arms import semantic_difference_column
from mp_retrieval.m3b_features import DIM, EDGE_ATTR, FAMILIES, MAX_SEEDS

PROJECTION_DIM = 64
N_EDGE_FEATURES = len(FAMILIES) + len(EDGE_ATTR)
FAMILY_INDEX = {f: i for i, f in enumerate(FAMILIES)}


@dataclass
class PackedBatch:
    """B queries packed along the node axis."""

    x: torch.Tensor            # (N, F) raw scalars, float32
    qptr: torch.Tensor         # (B + 1,) int64 node offsets
    node_query: torch.Tensor   # (N,) int64 query index per node
    emb: torch.Tensor          # (N, DIM) node embeddings
    qemb: torch.Tensor         # (B, DIM) query embeddings
    seedw: torch.Tensor        # (N, MAX_SEEDS) reach weights
    seed_nodes: torch.Tensor   # (B, MAX_SEEDS) global node index of each seed, -1 padded
    edge_index: torch.Tensor   # (2, M) global node indices, u -> v
    edge_attr: torch.Tensor    # (M, N_EDGE_FEATURES)
    gold: torch.Tensor         # (N,) bool

    @property
    def n_queries(self) -> int:
        return int(self.qptr.numel() - 1)


# ── per-query standardisation and segment softmax ───────────────────────────


def segment_zscore(x: torch.Tensor, node_query: torch.Tensor, n_queries: int, eps: float = 1e-6) -> torch.Tensor:
    """Within-query z-score of every column; a column constant within a query gives 0."""
    ones = torch.ones(node_query.numel(), dtype=x.dtype, device=x.device)
    counts = torch.zeros(n_queries, dtype=x.dtype, device=x.device).index_add_(0, node_query, ones).clamp_min(1.0).unsqueeze(1)
    mean = torch.zeros(n_queries, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, node_query, x) / counts
    centred = x - mean[node_query]
    var = torch.zeros(n_queries, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, node_query, centred * centred) / counts
    std = var.sqrt()[node_query]
    z = centred / std.clamp_min(eps)
    return torch.where(std < eps, torch.zeros_like(z), z)


def segment_log_softmax(scores: torch.Tensor, node_query: torch.Tensor, n_queries: int) -> torch.Tensor:
    smax = torch.full((n_queries,), -float("inf"), dtype=scores.dtype, device=scores.device)
    smax = smax.scatter_reduce(0, node_query, scores, reduce="amax", include_self=True)
    shifted = scores - smax[node_query]
    denom = torch.zeros(n_queries, dtype=scores.dtype, device=scores.device).index_add_(0, node_query, shifted.exp())
    return shifted - denom.clamp_min(1e-30).log()[node_query]


def listwise_loss(scores: torch.Tensor, batch: PackedBatch) -> torch.Tensor:
    """The historical listwise loss: per query, minus the mean log-softmax of the
    in-pool gold; queries with no in-pool gold contribute nothing."""
    B = batch.n_queries
    logp = segment_log_softmax(scores, batch.node_query, B)
    gold = batch.gold.to(scores.dtype)
    n_gold = torch.zeros(B, dtype=scores.dtype, device=scores.device).index_add_(0, batch.node_query, gold)
    per_query = -torch.zeros(B, dtype=scores.dtype, device=scores.device).index_add_(0, batch.node_query, logp * gold)
    has = n_gold > 0
    if not bool(has.any()):
        raise RuntimeError("training batch has no in-pool gold")
    return (per_query[has] / n_gold[has]).mean()


# ── the shared input block ───────────────────────────────────────────────────


def _project(linear: nn.Linear, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = Fn.gelu(linear(rows))
    return raw, Fn.normalize(raw, dim=-1)


def segment_mean_rows(values: torch.Tensor, index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros(n, values.shape[1], dtype=values.dtype, device=values.device).index_add_(0, index, values)
    ones = torch.ones(index.numel(), dtype=values.dtype, device=values.device)
    count = torch.zeros(n, dtype=values.dtype, device=values.device).index_add_(0, index, ones)
    return out / count.clamp_min(1.0).unsqueeze(1), count


class InputBlock(nn.Module):
    """[scalars raw and z-scored | ProjectionSemanticHead 258 | semantic_difference 1 |
    C prototype products 3 x 64 | D reach-prototype product 64] -> Linear -> H."""

    def __init__(self, n_scalars: int, hidden: int, base_index: int, dropout: float = 0.2):
        super().__init__()
        self.n_scalars = int(n_scalars)
        self.base_index = int(base_index)
        self.semantic = ProjectionSemanticHead(DIM, PROJECTION_DIM)
        self.difference_weight = nn.Parameter(torch.zeros(DIM))
        self.input_width = 2 * n_scalars + len(self.semantic.feature_names) + 1 + PROJECTION_DIM * (len(FAMILIES) + 1)
        self.linear = nn.Linear(self.input_width, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: PackedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        # Both projections are bias-free linear maps, so the projection of a mean is the
        # mean of the projections: every prototype below is aggregated in the 64-wide
        # projected space instead of the 1536-wide embedding space. Same function, same
        # gradients; the edge-wide gathers shrink 24x (a systems choice, not a model change).
        B = batch.n_queries
        N = batch.x.shape[0]
        P = self.semantic.projection_dim
        z = segment_zscore(batch.x, batch.node_query, B)
        base_z = z[:, self.base_index]
        pre_q = self.semantic.query_projection(batch.qemb)[batch.node_query]      # == query_projection(qemb[node_query])
        raw_q = Fn.gelu(pre_q)
        q_state = Fn.normalize(raw_q, dim=-1)
        pre_n = self.semantic.node_projection(batch.emb)
        raw_n = Fn.gelu(pre_n)
        n_state = Fn.normalize(raw_n, dim=-1)
        semantic = torch.cat([
            q_state, n_state, q_state * n_state, (q_state - n_state).abs(),
            (q_state * n_state).sum(-1, keepdim=True), (raw_q * raw_n).sum(-1, keepdim=True) / raw_n.shape[-1] ** 0.5,
        ], dim=-1)
        difference = semantic_difference_column(batch.qemb[batch.node_query], batch.emb, self.difference_weight).unsqueeze(1)
        parts = [batch.x, z, semantic, difference]
        # C: fixed neighbour prototypes per family, projected strictly after the aggregation
        # (node_projection(mean_u e_u) == mean_u node_projection(e_u), the map being linear)
        has_edges = batch.edge_attr.shape[0] > 0
        fam_of_edge = batch.edge_attr[:, : len(FAMILIES)].argmax(dim=1) if has_edges else None
        for f_i in range(len(FAMILIES)):
            if fam_of_edge is None:
                parts.append(torch.zeros(N, P, dtype=batch.x.dtype, device=batch.x.device))
                continue
            sel = fam_of_edge == f_i
            u, v = batch.edge_index[0, sel], batch.edge_index[1, sel]
            proto_pre, count = segment_mean_rows(pre_n[u], v, N)
            p_state = Fn.normalize(Fn.gelu(proto_pre), dim=-1)
            p_state = torch.where((count > 0).unsqueeze(1), p_state, torch.zeros_like(p_state))
            parts.append(p_state * q_state)
        # D: the 1/dist-weighted prototype of the seeds within two hops, projected after the
        # aggregation (an absent seed slot is the zero vector, whose image is zero)
        seed_pre = torch.zeros(B, MAX_SEEDS, P, dtype=pre_n.dtype, device=pre_n.device)
        valid = batch.seed_nodes >= 0
        seed_pre[valid] = pre_n[batch.seed_nodes[valid]]
        w = batch.seedw
        reach_pre = torch.einsum("ns,nsp->np", w, seed_pre[batch.node_query]) / w.sum(1, keepdim=True).clamp_min(1e-12)
        r_state = Fn.normalize(Fn.gelu(reach_pre), dim=-1)
        r_state = torch.where((w.sum(1) > 0).unsqueeze(1), r_state, torch.zeros_like(r_state))
        parts.append(r_state * n_state)
        h = self.dropout(Fn.gelu(self.linear(torch.cat(parts, dim=-1))))
        return h, base_z


class ResidualReadout(nn.Module):
    """score = w_b * base_z + Linear_out(h); Linear_out zero-initialised, w_b = 1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.base_weight = nn.Parameter(torch.ones(1))

    def forward(self, h: torch.Tensor, base_z: torch.Tensor) -> torch.Tensor:
        return self.base_weight * base_z + self.out(h).squeeze(-1)


# ── the arms ─────────────────────────────────────────────────────────────────


class QLSU(nn.Module):
    """qls_u_sota_v1: input block -> 2 hidden layers -> residual readout. No edges."""

    arm = "qls_u_sota_v1"

    def __init__(self, n_scalars: int, hidden: int, base_index: int, dropout: float = 0.2):
        super().__init__()
        self.hidden = int(hidden)
        self.input = InputBlock(n_scalars, hidden, base_index, dropout)
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.readout = ResidualReadout(hidden)

    def forward(self, batch: PackedBatch) -> torch.Tensor:
        h, base_z = self.input(batch)
        return self.readout(self.body(h), base_z)


class UniversalGAT(nn.Module):
    """gat_universal_v1 / gat_no_mp_v1: input block -> L x [GATv2Conv over the pool
    graph, residual, LayerNorm, GELU, dropout] -> residual readout. With
    ``message_passing=False`` the layers receive no edges: the self-loop that
    GATv2Conv adds is the only message, which is the control."""

    def __init__(self, n_scalars: int, hidden: int, base_index: int, layers: int = 2, heads: int = 4,
                 dropout: float = 0.2, message_passing: bool = True, families: tuple[str, ...] = FAMILIES):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.arm = "gat_universal_v1" if message_passing else "gat_no_mp_v1"
        self.hidden, self.layers, self.message_passing = int(hidden), int(layers), bool(message_passing)
        self.family_mask = torch.tensor([f in families for f in FAMILIES])
        self.input = InputBlock(n_scalars, hidden, base_index, dropout)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=N_EDGE_FEATURES,
                                              add_self_loops=True, fill_value=0.0, dropout=0.0) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = ResidualReadout(hidden)

    def forward(self, batch: PackedBatch) -> torch.Tensor:
        h, base_z = self.input(batch)
        if self.message_passing and batch.edge_attr.shape[0]:
            fam = batch.edge_attr[:, : len(FAMILIES)].argmax(dim=1)
            keep = self.family_mask.to(fam.device)[fam]
            edge_index, edge_attr = batch.edge_index[:, keep], batch.edge_attr[keep]
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=h.device)
            edge_attr = torch.empty(0, N_EDGE_FEATURES, dtype=h.dtype, device=h.device)
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index, edge_attr)
            h = self.dropout(Fn.gelu(norm(h + m)))
        return self.readout(h, base_z)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
