"""Semantic frontier S0-S3: how cheaply can a query and a candidate be compared?

QLS-v1 spends a learned ``768 -> 64`` projection on `q` and on `d` -- 98,304
parameters, 46% of the model, committed before a single structural feature is
read. This module asks whether that is necessary or merely conventional, by
carrying four rungs of decreasing semantic capacity and letting the measurement
decide.

The rungs and the five features are frozen in
``docs/QLS_V2_FEATURE_CATALOG.md`` (Group F, "the semantic frontier"). Nothing
here reads a graph, which is why this module is unblocked while the substrate
audit that gates the *structural* formulas is still running.

The interesting rung is S3. Its two learned scalars are not merely a cheaper
approximation of the v1 projection:

* ``semantic_product`` is the diagonal restriction of the general bilinear form
  ``q^T W d`` that the rank-64 projection approximates, so it is strictly less
  expressive there.
* ``semantic_difference`` is a learned weighted L1. ``|q - d|`` is not a
  bilinear function of `q` and `d`, so no projection in v1 could express it.
  This is new capacity, not only cheaper capacity.

**The initialization is part of the frozen design, not a free choice.** With
``w_i = 1/768`` and ``v_i = 0``, F4 is exactly the mean elementwise product -- a
monotone function of the raw dot product already available at S2 -- and F5 is
identically zero. S3 therefore *starts* as S2 plus one redundant channel and can
only depart from it by learning. That makes an S3-over-S2 gain attributable to
the learned weights rather than to the extra channels existing at all.
"""

from __future__ import annotations

import torch
from torch import nn

EMBEDDING_DIM = 768
EPS = 1e-12

PARAMETER_FREE_FEATURE_NAMES = (
    "cosine_qd",
    "dot_qd_pct",
    "mean_abs_diff",
)
LEARNED_FEATURE_NAMES = (
    "semantic_product",
    "semantic_difference",
)
SEMANTIC_FEATURE_NAMES = PARAMETER_FREE_FEATURE_NAMES + LEARNED_FEATURE_NAMES

# S0 is the honest floor: it asks how much of the result is semantic *at all*,
# the same way R0 asks how much is topological at all. Group A (retrieval ranks
# and RRF) is always present and is not a member of this group.
RUNG_FEATURES: dict[str, tuple[str, ...]] = {
    "S0": (),
    "S1": ("cosine_qd",),
    "S2": ("cosine_qd", "dot_qd_pct", "mean_abs_diff"),
    "S3": SEMANTIC_FEATURE_NAMES,
}
RUNGS = tuple(RUNG_FEATURES)

# Only F4 and F5 carry weights, one vector each over the embedding dimension.
RUNG_PARAMETERS: dict[str, int] = {
    "S0": 0,
    "S1": 0,
    "S2": 0,
    "S3": 2 * EMBEDDING_DIM,
}

# What v1 spends on the two rank-64 projections, for the reduction factor.
V1_SEMANTIC_PARAMETERS = 2 * EMBEDDING_DIM * 64


def within_query_percentile(values: torch.Tensor) -> torch.Tensor:
    """Rank of each value among the candidates of one query, scaled to [0, 1].

    Ties take the average of the ranks they span. The catalog fixes the
    quantity -- "within-query percentile of <q, d>" -- but not the tie rule, and
    the choice is not cosmetic: ordinal ranking through ``argsort`` (the
    convention in :mod:`l2_features`) resolves ties by position, so permuting a
    candidate list would change the feature. Averaging makes the value a
    function of the multiset alone.

    A single candidate is the degenerate all-tied case and scores 0.5, which is
    the limit of the tied-group rule rather than a separate convention: when all
    `n` values tie, the average rank is ``(n-1)/2`` and the percentile is 0.5 for
    every `n`.
    """
    count = values.numel()
    if count <= 1:
        return torch.full_like(values, 0.5)

    order = torch.argsort(values)
    ordinal = torch.empty(count, dtype=values.dtype, device=values.device)
    ordinal[order] = torch.arange(count, dtype=values.dtype, device=values.device)

    # Average the ordinal ranks within each group of exactly-equal values.
    unique, inverse = torch.unique(values, return_inverse=True)
    totals = torch.zeros(unique.numel(), dtype=values.dtype, device=values.device)
    totals.scatter_add_(0, inverse, ordinal)
    sizes = torch.zeros(unique.numel(), dtype=values.dtype, device=values.device)
    sizes.scatter_add_(0, inverse, torch.ones_like(ordinal))
    averaged = (totals / sizes)[inverse]

    return averaged / (count - 1)


def parameter_free_scalars(
    query: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """F1-F3 for one query against its candidate set.

    ``query`` is ``(dim,)``, ``candidates`` is ``(n, dim)``; returns ``(n, 3)``
    ordered as :data:`PARAMETER_FREE_FEATURE_NAMES`. No parameters, no graph.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be (n, dim); got {tuple(candidates.shape)}")
    if query.shape != candidates.shape[1:]:
        raise ValueError(
            f"query dim {tuple(query.shape)} does not match candidate dim "
            f"{tuple(candidates.shape[1:])}"
        )

    unit_query = query / query.norm().clamp_min(EPS)
    unit_candidates = candidates / candidates.norm(dim=1, keepdim=True).clamp_min(EPS)

    cosine = unit_candidates @ unit_query
    dot_percentile = within_query_percentile(candidates @ query)
    mean_abs_diff = (candidates - query).abs().mean(dim=1)

    return torch.stack([cosine, dot_percentile, mean_abs_diff], dim=1)


class SemanticHead(nn.Module):
    """The semantic branch at a chosen rung.

    Below S3 this module holds no parameters at all, which is the point of the
    frontier rather than an optimization: S0-S2 test whether a learned semantic
    comparison is needed before one is paid for.
    """

    def __init__(self, rung: str = "S3", dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        if rung not in RUNG_FEATURES:
            raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
        self.rung = rung
        self.dim = dim
        self.feature_names = RUNG_FEATURES[rung]

        if rung == "S3":
            # w_i = 1/dim makes F4 the mean elementwise product at
            # initialization; v_i = 0 makes F5 inert but still trainable,
            # since dF5/dv_i = |q_i - d_i| does not vanish.
            self.product_weight = nn.Parameter(torch.full((dim,), 1.0 / dim))
            self.difference_weight = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("product_weight", None)
            self.register_parameter("difference_weight", None)

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Return ``(n, len(self.feature_names))`` for one query."""
        if not self.feature_names:
            return candidates.new_zeros((candidates.shape[0], 0))

        scalars = parameter_free_scalars(query, candidates)
        columns = [
            scalars[:, PARAMETER_FREE_FEATURE_NAMES.index(name)]
            for name in self.feature_names
            if name in PARAMETER_FREE_FEATURE_NAMES
        ]

        if self.rung == "S3":
            columns.append((candidates * query) @ self.product_weight)
            columns.append((candidates - query).abs() @ self.difference_weight)

        return torch.stack(columns, dim=1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
