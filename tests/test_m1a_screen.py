"""M1A's shared scorer and per-regime feature assembly.

Covers why this model is not ``ExplicitFeatureMLP`` (the semantic branch must
receive live gradients, so it cannot be baked into a precomputed column), and
that the R3-bounded context builder and NODE_ROLE column match
``candidate_expansion_v2``/``overlap_audit`` bit-exactly rather than
reimplementing their arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, ExpansionBudget, expand
from mp_retrieval.m1a_screen import (
    FAMILY_COLUMNS,
    HEAD_WIDTH,
    M1AScorer,
    base_columns,
    node_role_column,
    r3_bounded_context,
)
from mp_retrieval.structural_features import LOCAL_FEATURE_NAMES

# --- base_columns ---


def test_seed_identity_column_is_local10_column_0_unchanged():
    local10 = np.zeros((3, 10), dtype=np.float32)
    local10[:, 0] = [1.0, 0.0, 0.0]
    out = base_columns(local10, dense=np.array([7]), splade=np.array([9]), candidates=np.array([7, 9, 11]))
    assert out[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_agreement_is_and_of_dense_and_splade_membership():
    local10 = np.zeros((3, 10), dtype=np.float32)
    candidates = np.array([7, 9, 11])
    out = base_columns(local10, dense=np.array([7, 9]), splade=np.array([9, 11]), candidates=candidates)
    # 7: dense only, 9: both, 11: splade only.
    assert out[:, 3].tolist() == [0.0, 1.0, 0.0]


def test_r3_only_structural_candidate_gets_zero_retrieval_columns_not_an_error():
    """A64-admitted nodes were never retrieval-scored -- zero, not a raised error.

    ``candidates`` here is C3 = Cq u A64: node 7 is a normal Cq member (ranked
    by both retrievers), node 99 is A64-only and appears in neither ranking.
    dense/splade stay subsets of candidates either way -- Cq's own retrieval
    lists never grow under R3 -- so rank_feature_rows's "every ranked ID must
    be in the candidate union" invariant still holds; it is candidates that
    grows beyond the rankings, not the other way round. Unlike
    scripts/run_graph_context_d4.py's own builder, which additionally asserts
    every candidate is ranked by at least one retriever (true for R1/R2's Cq,
    false for R3's C3).
    """
    local10 = np.zeros((2, 10), dtype=np.float32)
    out = base_columns(local10, dense=np.array([7]), splade=np.array([7]), candidates=np.array([7, 99]))
    assert out[0, 1:].tolist() != [0.0, 0.0, 0.0]
    assert out[1, 1:].tolist() == [0.0, 0.0, 0.0]


def test_base_columns_rejects_a_row_count_mismatch():
    local10 = np.zeros((2, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        base_columns(local10, dense=np.array([7]), splade=np.array([7]), candidates=np.array([7, 9, 11]))


# --- family column ranges, checked against the real imported names ---


def test_family_columns_match_the_real_local_feature_names():
    assert LOCAL_FEATURE_NAMES[0] == "distance_0"  # SEED, in BASE, not repeated in FAMILY_COLUMNS
    assert LOCAL_FEATURE_NAMES[FAMILY_COLUMNS["GEOMETRY"]] == (
        "distance_1",
        "distance_2",
        "distance_3_plus_or_unreachable",
    )
    assert LOCAL_FEATURE_NAMES[FAMILY_COLUMNS["SUPPORT"]] == ("seed_connections",)
    assert LOCAL_FEATURE_NAMES[FAMILY_COLUMNS["PATH"]] == (
        "paths_length_1",
        "paths_length_2",
        "paths_length_3",
    )


# --- node_role_column ---


def test_node_role_is_zero_everywhere_under_r1_r2_where_scored_equals_cq():
    pool = np.array([7, 9])
    out = node_role_column(cq=pool, scored=pool, context=pool)
    assert out[:, 0].tolist() == [0.0, 0.0]


def test_node_role_marks_exactly_the_a64_admitted_rows_under_r3():
    cq = np.array([7, 9])
    scored = np.array([7, 9, 11])  # 11 = A64-admitted, beyond Cq
    context = np.array([7, 9, 11, 20])  # 20 = CONTEXT_ONLY, never scored
    out = node_role_column(cq=cq, scored=scored, context=context)
    assert out[:, 0].tolist() == [0.0, 0.0, 1.0]


def test_node_role_column_is_two_state_not_three_state():
    """CONTEXT_ONLY nodes are never scored, so they never appear in `scored` at all."""
    cq = np.array([7])
    scored = np.array([7, 11])
    context = np.array([7, 11, 20])
    out = node_role_column(cq=cq, scored=scored, context=context)
    assert out.shape == (2, 1)


# --- r3_bounded_context: must match candidate_expansion_v2.expand bit-exactly ---

NUM_NODES = 12
EDGES = ((0, 1), (0, 2), (0, 3), (0, 8), (0, 11), (1, 4), (2, 5), (3, 6), (7, 0))
POOL = np.array([0, 7, 9, 10], dtype=np.int64)
SEEDS = np.array([0], dtype=np.int64)
ANCHOR = 0


def _csr(edges, num_nodes: int):
    rows: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        rows[source].append(target)
    col: list[int] = []
    rowptr = [0]
    for row in rows:
        col.extend(sorted(row))
        rowptr.append(len(col))
    return np.asarray(rowptr, dtype=np.int64), np.asarray(col, dtype=np.int64)


def _embeddings() -> np.ndarray:
    return np.zeros((NUM_NODES, 3), dtype=np.float64)


def test_r3_bounded_context_admitted_set_matches_a_direct_expand_call():
    rowptr, col = _csr(EDGES, NUM_NODES)
    u2 = POOL  # a degenerate U2 == Cq is enough to isolate the A64/union arithmetic
    budget = ExpansionBudget()
    u3_bounded, c3, a64, expansion = r3_bounded_context(
        family_rowptr=rowptr, family_col=col, node_embeddings=_embeddings(),
        anchor=ANCHOR, pool=POOL, seeds=SEEDS, budget=budget, num_nodes=NUM_NODES, u2=u2,
    )
    direct = expand(
        STRUCTURAL, rowptr=rowptr, col=col, node_embeddings=_embeddings(), query_embedding=None,
        anchor=ANCHOR, pool=POOL, seeds=SEEDS, budget=budget, num_nodes=NUM_NODES,
    )
    assert a64.tolist() == direct.admitted.tolist() == expansion.admitted.tolist()
    assert c3.tolist() == direct.additive_pool.tolist()


def test_u3_bounded_is_the_stable_union_of_u2_and_a64_not_a_second_expansion():
    rowptr, col = _csr(EDGES, NUM_NODES)
    u2 = np.array([0, 7, 9, 10, 50], dtype=np.int64)  # a node outside the expansion's own reach
    budget = ExpansionBudget()
    u3_bounded, c3, a64, _expansion = r3_bounded_context(
        family_rowptr=rowptr, family_col=col, node_embeddings=_embeddings(),
        anchor=ANCHOR, pool=POOL, seeds=SEEDS, budget=budget, num_nodes=NUM_NODES, u2=u2,
    )
    assert u3_bounded.tolist() == sorted(set(u2.tolist()) | set(a64.tolist()))
    assert set(u2.tolist()) <= set(u3_bounded.tolist())
    assert set(c3.tolist()) <= set(u3_bounded.tolist())


# --- M1AScorer ---


TEST_EMBEDDING_DIM = 8


def _model(*, precomputed_width=4, semantic_rung="S3", dropout=0.0, temperature=1.0):
    return M1AScorer(
        precomputed_width=precomputed_width,
        semantic_rung=semantic_rung,
        dropout=dropout,
        temperature=temperature,
        embedding_dim=TEST_EMBEDDING_DIM,
    )


def _forward_inputs(*, num_queries=2, per_query=3, precomputed_width=4, seed=0):
    torch.manual_seed(seed)
    total = num_queries * per_query
    nodes = torch.randn(total, TEST_EMBEDDING_DIM)
    queries = torch.randn(num_queries, TEST_EMBEDDING_DIM)
    batch_index = torch.repeat_interleave(torch.arange(num_queries), per_query)
    structural = torch.randn(total, precomputed_width) if precomputed_width else None
    return nodes, queries, batch_index, structural


def test_semantic_head_param_counts_match_the_rung_live():
    assert _model(semantic_rung="S2").semantic_parameter_count() == 0
    assert _model(semantic_rung="S3").semantic_parameter_count() == 2 * TEST_EMBEDDING_DIM


def test_forward_explicit_output_shape_matches_total_candidates():
    model = _model()
    nodes, queries, batch_index, structural = _forward_inputs(precomputed_width=4)
    scores = model.forward_explicit(nodes, queries, batch_index, structural)
    assert scores.shape == (nodes.shape[0],)


def test_forward_explicit_rejects_a_precomputed_width_mismatch():
    model = _model()
    nodes, queries, batch_index, _structural = _forward_inputs(precomputed_width=4)
    wrong = torch.randn(nodes.shape[0], 3)
    with pytest.raises(ValueError):
        model.forward_explicit(nodes, queries, batch_index, wrong)


def test_forward_explicit_works_from_semantic_columns_alone():
    model = _model(precomputed_width=0)
    nodes, queries, batch_index, _structural = _forward_inputs(precomputed_width=0)
    scores = model.forward_explicit(nodes, queries, batch_index, None)
    assert scores.shape == (nodes.shape[0],)


def test_s3_semantic_parameters_receive_real_gradients_every_step():
    """The whole reason this model exists instead of a precomputed column.

    If semantic_product/semantic_difference's weights never saw a gradient,
    S3's 1,536 learned parameters would sit at their random initial value
    forever -- indistinguishable, in the fitted model, from not being trained
    at all.
    """
    model = _model()
    nodes, queries, batch_index, structural = _forward_inputs(precomputed_width=4)
    scores = model.forward_explicit(nodes, queries, batch_index, structural)
    scores.sum().backward()
    assert model.semantic_head.product_weight.grad is not None
    assert model.semantic_head.difference_weight.grad is not None
    assert torch.any(model.semantic_head.product_weight.grad != 0.0)


def test_s2_control_contributes_no_trainable_parameters_to_the_model():
    model = _model(semantic_rung="S2")
    assert model.semantic_parameter_count() == 0
    assert model.trainable_parameter_count() == model.scorer_parameter_count()


def test_head_width_is_the_declared_fixed_constant():
    model = _model()
    first_layer = model.scorer[0]
    assert first_layer.out_features == HEAD_WIDTH == 32
