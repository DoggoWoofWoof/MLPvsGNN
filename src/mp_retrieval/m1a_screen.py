"""M1A's shared scorer and per-regime, per-arm feature assembly.

Not ``ExplicitFeatureMLP``: that model's ``forward_explicit`` always applies
its own learned 768->64 node/query projection (98,304 params -- the exact
projection ``qls_v2_semantic.py``'s own module docstring names as the thing
the S0-S3 semantic ladder exists to question) before any structural features
are concatenated. Reusing it here would put a fixed, always-on 98,304-param
block in every arm regardless of ``semantic_rung``, confounding the very
S2-vs-S3 contrast ``configs/m1a_feature_screen.yaml#base.semantic_rung``
exists to keep legible. See that file's ``model_architecture`` key for the
full rationale.

``M1AScorer`` has no projection of its own: it calls ``SemanticHead`` live,
per query, on raw embeddings (so its learned parameters receive real
gradients every step) and concatenates that with precomputed BASE/family/
NODE_ROLE columns before one small scoring head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .candidate_expansion_v2 import STRUCTURAL, Expansion, ExpansionBudget, expand
from .linear_control import rank_feature_rows
from .overlap_audit import ROLE_STRUCTURAL_SCORED_CANDIDATE, node_roles
from .qls_v2_semantic import EMBEDDING_DIM, SemanticHead
from .structural_features import LOCAL_FEATURE_NAMES

#: (K+1)/(K+rank+1), frozen for dense/SPLADE reciprocal rank -- see
#: linear_control.py and scripts/run_graph_context_d4.py.
RETRIEVAL_RANK_CONSTANT = 60

#: A fixed, small, round width -- not parameter-matched to the historical
#: ~213K budget, since that budget is dominated by the projection this
#: architecture deliberately drops. See configs/m1a_feature_screen.yaml
#: #model_architecture.head_width for the full rationale.
HEAD_WIDTH = 32

#: Column ranges into qls_local_features' frozen 10-column output
#: (structural_features.LOCAL_FEATURE_NAMES). Column 0 (distance_0) is SEED
#: and is part of BASE separately, via base_columns below -- not repeated
#: here. DIFFUSION (8) and TOPOLOGY (9) are excluded from M1A's matrix.
FAMILY_COLUMNS: dict[str, slice] = {
    "GEOMETRY": slice(1, 4),
    "SUPPORT": slice(4, 5),
    "PATH": slice(5, 8),
}
if LOCAL_FEATURE_NAMES[0] != "distance_0":
    raise AssertionError("LOCAL_FEATURE_NAMES column 0 must be distance_0 (SEED)")


def base_columns(
    local10: np.ndarray,
    dense: np.ndarray,
    splade: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """BASE's four precomputed columns: seed_identity, dense_rr, splade_rr, agreement.

    ``local10`` is ``qls_local_features``'s ten columns, already reindexed to
    ``candidates`` order (row i describes ``candidates[i]``). Unlike
    ``scripts/run_graph_context_d4.py``'s own retriever-agreement builder,
    this does not assert every candidate is ranked by at least one
    retriever: that invariant holds for R1/R2's ``Cq`` by construction but
    not for R3's ``C3 = Cq ∪ A64`` -- A64-admitted nodes are genuinely
    unranked, not merely low-ranked (see the R3-only-structural-candidates
    note in ``configs/m1a_feature_screen.yaml#base``).
    """

    local10 = np.asarray(local10, dtype=np.float32)
    if local10.shape[0] != len(candidates):
        raise ValueError(
            f"local10 has {local10.shape[0]} rows for {len(candidates)} candidates"
        )
    seed_identity = local10[:, 0:1]
    ranks = rank_feature_rows(dense, splade, candidates, constant=RETRIEVAL_RANK_CONSTANT)
    in_dense = ranks[:, 0] > 0.0
    in_splade = ranks[:, 1] > 0.0
    agreement = (in_dense & in_splade).astype(np.float32).reshape(-1, 1)
    return np.concatenate([seed_identity, ranks.astype(np.float32), agreement], axis=1)


def node_role_column(*, cq: np.ndarray, scored: np.ndarray, context: np.ndarray) -> np.ndarray:
    """``is_structurally_admitted``: 1.0 for STRUCTURAL_SCORED_CANDIDATE, 0.0 for RETRIEVAL_CANDIDATE.

    One row per entry of ``scored``, in ``scored``'s own order -- matching
    how every other column here is indexed. Exposes
    ``overlap_audit.node_roles``'s already-proven partition as a training
    column; see ``configs/m1a_feature_screen.yaml
    #feature_catalog.families.NODE_ROLE``.
    """

    roles = node_roles(cq=cq, scored=scored, context=context)
    admitted = np.asarray(roles[ROLE_STRUCTURAL_SCORED_CANDIDATE], dtype=np.int64)
    admitted_set = set(admitted.tolist())
    scored = np.asarray(scored, dtype=np.int64)
    return np.asarray(
        [[1.0 if int(node_id) in admitted_set else 0.0] for node_id in scored],
        dtype=np.float32,
    )


def r3_bounded_context(
    *,
    family_rowptr: np.ndarray,
    family_col: np.ndarray,
    node_embeddings,
    anchor: int,
    pool: np.ndarray,
    seeds: np.ndarray,
    budget: ExpansionBudget,
    num_nodes: int,
    u2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Expansion]:
    """One query's ``(U3_bounded, C3, A64)``, mirroring ``run_m0c_bounded_r3.py`` exactly.

    ``U3_bounded = stable_union(U2, A64)``, ``C3 = Cq ∪ A64``. No second
    ``TARGET_H1`` walk -- every structurally-scoreable node is admitted into
    the domain without re-expanding around it. Returns the ``Expansion``
    itself too, since callers need ``expansion.admitted`` for
    ``regime_set_invariants``-style checks the way M0C's own runner does.
    """

    expansion = expand(
        STRUCTURAL,
        rowptr=family_rowptr,
        col=family_col,
        node_embeddings=node_embeddings,
        query_embedding=None,
        anchor=anchor,
        pool=pool,
        seeds=seeds,
        budget=budget,
        num_nodes=num_nodes,
    )
    c3 = expansion.additive_pool
    a64 = expansion.admitted
    u3_bounded = np.union1d(np.asarray(u2, dtype=np.int64), a64)
    return u3_bounded, c3, a64, expansion


class M1AScorer(nn.Module):
    """BASE + arm columns (precomputed) concatenated with a live semantic branch, scored by one small MLP.

    ``forward_explicit`` matches ``ExplicitFeatureMLP``'s own signature so
    ``scripts/run_sa_mlp_confirmation.py``'s ``_fit``/``_score_once``/
    ``_prepare_batch`` shell drives this model unmodified under the literal
    ``model_name="sa_mlp"``.
    """

    def __init__(
        self,
        *,
        precomputed_width: int,
        semantic_rung: str,
        dropout: float,
        temperature: float,
        head_width: int = HEAD_WIDTH,
        embedding_dim: int = EMBEDDING_DIM,
        semantic_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # M2B varies the semantic branch and nothing else, so it injects its own
        # head rather than getting a second scorer. Default None keeps every
        # M1A/M1B/M2 construction byte-identical: the branch below is the only
        # one those callers reach.
        if semantic_head is None:
            semantic_head = SemanticHead(rung=semantic_rung, dim=embedding_dim)
        elif getattr(semantic_head, "rung", None) != semantic_rung:
            raise ValueError(
                f"semantic_head is rung {getattr(semantic_head, 'rung', None)!r} but the model "
                f"was asked for {semantic_rung!r}; a fit recorded under the wrong rung name is "
                "unrecoverable after the fact"
            )
        self.semantic_head = semantic_head
        self.precomputed_width = int(precomputed_width)
        input_dim = self.precomputed_width + len(self.semantic_head.feature_names)
        if input_dim <= 0:
            raise ValueError("M1AScorer needs at least one input column")
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, head_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, 1),
        )
        self.temperature = float(temperature)

    def forward_explicit(
        self,
        nodes: torch.Tensor,
        queries: torch.Tensor,
        batch_index: torch.Tensor,
        structural_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if structural_features is not None and structural_features.shape[1] != self.precomputed_width:
            raise ValueError(
                f"expected {self.precomputed_width} precomputed columns, "
                f"got {tuple(structural_features.shape)}"
            )
        if self.semantic_head.feature_names:
            blocks = []
            for position in range(queries.shape[0]):
                mask = batch_index == position
                blocks.append(self.semantic_head(queries[position], nodes[mask]))
            semantic = torch.cat(blocks, dim=0)
        else:
            semantic = nodes.new_zeros((nodes.shape[0], 0))
        pieces = [
            piece
            for piece in (structural_features, semantic)
            if piece is not None and piece.shape[1] > 0
        ]
        if not pieces:
            raise ValueError("M1AScorer received no input columns at all")
        combined = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
        return self.scorer(combined).squeeze(-1) / self.temperature

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def semantic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.semantic_head.parameters() if p.requires_grad)

    def scorer_parameter_count(self) -> int:
        return sum(p.numel() for p in self.scorer.parameters() if p.requires_grad)


def build_m1a_model(
    *,
    precomputed_width: int,
    semantic_rung: str,
    dropout: float,
    temperature: float,
    head_width: int = HEAD_WIDTH,
    embedding_dim: int = EMBEDDING_DIM,
    semantic_head: nn.Module | None = None,
) -> M1AScorer:
    return M1AScorer(
        precomputed_width=precomputed_width,
        semantic_rung=semantic_rung,
        dropout=dropout,
        temperature=temperature,
        head_width=head_width,
        embedding_dim=embedding_dim,
        semantic_head=semantic_head,
    )
