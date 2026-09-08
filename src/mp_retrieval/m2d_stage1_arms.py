"""M2D Stage 1's two arms: S4's branch, plus one thing each.

Section 8b reduced the repair ladder to a pair. Both arms are the same object
as native S4 with columns appended, and the pair is a causal control:

* **A1** appends ``cosine_qd`` and ``mean_abs_diff`` -- raw-embedding access
  with no parameters at all. It asks whether *seeing the 1536-d payload* is
  what matters.
* **A3-MINIMAL** appends ``semantic_difference``, S3's learned weighted L1 over
  the same 1536 dims. It asks whether the *learned dimension-wise weighting* is
  what matters.

A1's ``mean_abs_diff`` is exactly A3-MINIMAL's column at the fixed weight
``1/dim``, so the two differ on that column only in whether 1,536 weights are
free. The nesting is partial, not clean -- A1 also carries ``cosine_qd``, which
A3-MINIMAL does not -- and the declaration says so rather than claiming a
tidier relationship than exists.

Composition, not reimplementation
---------------------------------
Neither class re-derives S4. Each *holds* a live
:class:`~mp_retrieval.m2b_semantic_control.ProjectionSemanticHead` as a
submodule and concatenates its output. So "all existing S4 computation" in
section 10's systems contract is included by construction: it is the same
object M2B fit, with the same parameters, the same GELU, the same
normalization and the same 258 columns, not a copy that could drift.

The added columns are a different matter, and the choice made here should be
visible rather than buried.

The live expressions live *inside* larger functions.
``semantic_difference`` is one line of :meth:`SemanticHead.forward`, which for
S3 also computes ``semantic_product`` and, before either, calls
``parameter_free_scalars`` -- and that computes ``dot_qd_pct`` through
``within_query_percentile``, an argsort, a unique and two scatter-adds.
``cosine_qd`` and ``mean_abs_diff`` are two of the three columns that same
function returns. Neither arm has ``dot_qd_pct`` or ``semantic_product``.

So there were two ways to get one implementation, and both were worse:

*Call the live functions and discard the extra columns.* Then the p95 that
section 10 requires would include the cost of columns the model does not have,
and the number would describe a different object than the one being judged.
The gate applies the systems veto to both arms, so an inflated A1 could fail
for a reason that is not A1.

*Extract the expressions into shared functions.* That edits
``qls_v2_semantic.py`` -- frozen upstream science -- and the motive would be a
latency benchmark. "Systems convenience never edits science" is a standing rule
on this track, and this is exactly the case it names.

What is left is transcription, which the declaration permits and constrains:
the formula is recovered from the code, never rewritten from prose. Two
implementations of one formula can silently diverge, so the divergence is made
loud instead of promised away. ``tests/test_m2d_stage1_arms.py`` asserts exact
equality -- ``torch.equal``, not ``allclose`` -- against the live modules
across shapes, seeds and degenerate cases, and separately asserts that the live
source still contains the expression these functions were copied from. Editing
the live formula turns that red.
"""

from __future__ import annotations

import torch
from torch import nn

from .m2b_semantic_control import PROJECTION_DIM, ProjectionSemanticHead
from .qls_v2_semantic import EPS

#: The arm names section 8b froze. They are also the ``rung`` each head
#: reports, so ``semantic_rung_fingerprint`` gives an arm fit a provenance
#: hash that no S2, S3 or S4 fit can collide with.
CONTROL_ARM = "A1"
CANDIDATE_ARM = "A3_MINIMAL"

#: A1's two columns, in the order it appends them. Both names are the live
#: ones from :data:`PARAMETER_FREE_FEATURE_NAMES`, so a downstream reader
#: sees the same identifier for the same quantity.
RAW_SCALAR_NAMES = ("cosine_qd", "mean_abs_diff")

#: A3-MINIMAL's one column, named as S3 names it.
DIFFERENCE_NAME = "semantic_difference"


def raw_embedding_scalars(
    query: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """``cosine_qd`` and ``mean_abs_diff`` for one query, as ``(n, 2)``.

    Transcribed from ``parameter_free_scalars``, minus its middle column.
    ``dot_qd_pct`` is not omitted for speed: section 7 of the declaration
    excludes it from A1 by a rule frozen before any Stage-0 number existed, so
    computing it would be computing a column this arm does not have.
    """

    unit_query = query / query.norm().clamp_min(EPS)
    unit_candidates = candidates / candidates.norm(dim=1, keepdim=True).clamp_min(EPS)

    cosine = unit_candidates @ unit_query
    mean_abs_diff = (candidates - query).abs().mean(dim=1)

    return torch.stack([cosine, mean_abs_diff], dim=1)


def semantic_difference_column(
    query: torch.Tensor, candidates: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """S3's ``f(q, d) = sum_i v_i * |q_i - d_i|``, as ``(n,)``.

    Transcribed from the S3 branch of :meth:`SemanticHead.forward`. The whole
    formula is the one expression; there is no reduction, no bias and no
    normalization around it, and adding any would be a different feature than
    the one Stage 0 found signal in.
    """

    return (candidates - query).abs() @ weight


class _ProjectionArm(nn.Module):
    """What both arms share: a live S4 branch and columns appended to it.

    Interface-compatible with ``SemanticHead`` and ``ProjectionSemanticHead``
    -- ``rung``, ``dim``, ``feature_names``, ``forward``, ``parameter_count``
    -- because M2B's scorer width is derived from ``len(feature_names)`` and
    its runner switches on ``rung``. An arm that needed either of those changed
    would not be the same experiment with one column added.
    """

    added_names: tuple[str, ...] = ()

    def __init__(self, rung: str, dim: int, projection_dim: int = PROJECTION_DIM) -> None:
        super().__init__()
        self.rung = rung
        self.dim = int(dim)
        self.projection_dim = int(projection_dim)
        # The live S4 object, not a reimplementation of it.
        self.projection = ProjectionSemanticHead(dim=dim, projection_dim=projection_dim)
        self.feature_names = tuple(self.projection.feature_names) + self.added_names

    def added_columns(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # The projection runs first because its own shape validation is the
        # contract both live heads enforce; reusing it keeps one error message
        # for one mistake rather than a second copy that could drift from it.
        projected = self.projection(query, candidates)
        return torch.cat([projected, self.added_columns(query, candidates)], dim=-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def added_parameter_count(self) -> int:
        """What this arm spends beyond native S4. Reported, never assumed."""

        return self.parameter_count() - self.projection.parameter_count()


class RawSemanticSkipHead(_ProjectionArm):
    """A1: S4 plus raw-embedding access, and not one parameter more.

    The control. If A1 moves the blockers as far as A3-MINIMAL does, the story
    is that S4's 64-d bottleneck hides the payload, and the 1,536 learned
    weights are not what did the work -- so the simpler arm wins. That is
    section 8b's case 1, and it is a real possible outcome, not a formality.
    """

    added_names = RAW_SCALAR_NAMES

    def __init__(self, dim: int, projection_dim: int = PROJECTION_DIM) -> None:
        super().__init__(CONTROL_ARM, dim, projection_dim)

    def added_columns(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return raw_embedding_scalars(query, candidates)


class SemanticDifferenceHead(_ProjectionArm):
    """A3-MINIMAL: S4 plus S3's learned weighted L1 over the raw 1536 dims.

    The candidate, and one model -- not an ensemble, not S3 running beside S4.
    The parameter is S3's exactly: one vector over the embedding dimension,
    zero-initialized, no bias. Zero init makes the column inert at step 0 and
    still trainable, since ``d/dv_i = |q_i - d_i|`` does not vanish, so a gain
    is attributable to the weights being *learned* rather than to a channel
    existing.
    """

    added_names = (DIFFERENCE_NAME,)

    def __init__(self, dim: int, projection_dim: int = PROJECTION_DIM) -> None:
        super().__init__(CANDIDATE_ARM, dim, projection_dim)
        # S3's initialization, which the declaration records as part of the
        # frozen design rather than a free choice.
        self.difference_weight = nn.Parameter(torch.zeros(self.dim))

    def added_columns(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        column = semantic_difference_column(query, candidates, self.difference_weight)
        return column.unsqueeze(1)


def build_arm_head(arm: str, dim: int, projection_dim: int = PROJECTION_DIM) -> nn.Module:
    """The one place an arm name becomes a module.

    Deliberately shaped like ``run_m2b_semantic_minimality.build_semantic_head``:
    an unknown name raises rather than falling back to anything, so a typo in a
    launcher cannot quietly fit a different arm than the one it reports.
    """

    if arm == CONTROL_ARM:
        return RawSemanticSkipHead(dim=dim, projection_dim=projection_dim)
    if arm == CANDIDATE_ARM:
        return SemanticDifferenceHead(dim=dim, projection_dim=projection_dim)
    raise ValueError(
        f"unknown Stage-1 arm {arm!r}; section 8b declares {[CONTROL_ARM, CANDIDATE_ARM]}"
    )
