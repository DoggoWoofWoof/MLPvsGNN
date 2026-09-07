"""S4: the projection-style semantic control M2B measures S2 and S3 against.

QLS-v1 spent a learned ``embedding_dim -> 64`` projection on the query and on
the candidate and fed four blocks plus two scalars of their interaction into its
scorer. ``qls_v2_semantic`` records that cost as
:data:`~mp_retrieval.qls_v2_semantic.V1_SEMANTIC_PARAMETERS`, 98,304, and the
prose around this track has repeated that number for months.

**98,304 is a fact about a 768-dimensional payload, and this track's payload is
1536-dimensional.** The constant is computed from that module's
``EMBEDDING_DIM = 768`` default, while every M1A/M1B/M2 fit ran against node
embeddings the runner asserts are 1536 wide. Instantiating the same projection
against the real payload costs twice as much, and no amount of reading the old
number would have said so. So this module exists to be *instantiated and
counted*, never quoted: nothing here hardcodes a parameter total, and
``scripts/m2b_semantic_audit.py`` reports what a live instance actually holds.

What is and is not historical
-----------------------------
The semantic block below is formula-identical to
``operator_models.ExplicitFeatureMLP.forward_explicit``'s: the same two
bias-free projections, the same GELU, the same post-GELU normalization, the
same four blocks (query state, node state, elementwise product, absolute
difference) and the same two scalars (normalized dot, raw dot scaled by
``sqrt(projection_dim)``).

Two things are NOT historical, and calling the result "the v1 model" would be
wrong on both counts:

* the input width is the current payload's 1536, not v1's 768; and
* what consumes these columns is M2's frozen scorer, not v1's
  parameter-matched head.

configs/m2b_semantic_minimality.yaml therefore labels this rung
CURRENT_DIMENSION_PROJECTION_CONTROL rather than HISTORICAL_EXACT. It is a
control for "does a projection-style semantic branch buy anything over two
learned vectors", which is M2B's question. It is not a reconstruction of v1,
which would need v1's scorer too and would answer a different one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

#: v1's projection width. Kept as the one historical constant this module
#: carries, because the control is only a control if the width is v1's; the
#: parameter *count* that follows from it is derived, never asserted.
PROJECTION_DIM = 64

RUNG = "S4"

#: The four blocks are ``projection_dim`` wide each and the two scalars are one
#: each, so the branch is ``4 * projection_dim + 2`` columns. Named rather than
#: numbered because M2B's scorer width is derived from ``len(feature_names)``
#: and a bare integer there would be a second place to keep in step.
BLOCK_NAMES = ("query_state", "node_state", "state_product", "state_absolute_difference")
SCALAR_NAMES = ("normalized_state_dot", "raw_projection_dot_scaled")


def feature_names(projection_dim: int = PROJECTION_DIM) -> tuple[str, ...]:
    blocks = tuple(
        f"{name}[{index}]" for name in BLOCK_NAMES for index in range(projection_dim)
    )
    return blocks + SCALAR_NAMES


class ProjectionSemanticHead(nn.Module):
    """The v1 semantic branch, at whatever width the payload actually is.

    Interface-compatible with :class:`~mp_retrieval.qls_v2_semantic.SemanticHead`
    -- ``feature_names``, ``forward(query, candidates)`` returning
    ``(n, len(feature_names))``, and ``parameter_count()`` -- so M2B's model can
    take one or the other and change nothing else. That compatibility is the
    experiment: if the scorer, the structural columns and the training loop are
    the same object in all three arms, a difference between them is the semantic
    branch.
    """

    def __init__(self, dim: int, projection_dim: int = PROJECTION_DIM) -> None:
        super().__init__()
        if dim <= 0 or projection_dim <= 0:
            raise ValueError(f"dim and projection_dim must be positive; got {dim}, {projection_dim}")
        self.rung = RUNG
        self.dim = int(dim)
        self.projection_dim = int(projection_dim)
        self.feature_names = feature_names(self.projection_dim)
        # bias=False matches operator_models.OperatorModel, which is where the
        # historical projection lives. A bias here would be a quiet 128-parameter
        # departure from the thing this is supposed to control for.
        self.query_projection = nn.Linear(self.dim, self.projection_dim, bias=False)
        self.node_projection = nn.Linear(self.dim, self.projection_dim, bias=False)

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 2:
            raise ValueError(f"candidates must be (n, dim); got {tuple(candidates.shape)}")
        if query.shape != candidates.shape[1:]:
            raise ValueError(
                f"query dim {tuple(query.shape)} does not match candidate dim "
                f"{tuple(candidates.shape[1:])}"
            )

        raw_nodes = F.gelu(self.node_projection(candidates))
        raw_query = F.gelu(self.query_projection(query))
        node_state = F.normalize(raw_nodes, dim=-1)
        query_state = F.normalize(raw_query, dim=-1).expand_as(node_state)

        return torch.cat(
            [
                query_state,
                node_state,
                query_state * node_state,
                torch.abs(query_state - node_state),
                (query_state * node_state).sum(dim=-1, keepdim=True),
                (raw_query * raw_nodes).sum(dim=-1, keepdim=True)
                / raw_nodes.shape[-1] ** 0.5,
            ],
            dim=-1,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
