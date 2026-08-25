import torch

from mp_retrieval.operator_models import (
    COVERAGE_OFFSET_MODEL,
    SCREEN_MODELS,
    STRUCTURE_AWARE_MODELS,
    build_explicit_feature_mlp,
    build_operator_model,
    model_parameter_counts,
)


def _inputs():
    torch.manual_seed(3)
    nodes = torch.randn(7, 8)
    queries = torch.randn(2, 8)
    anchors = torch.randn(2, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edges = torch.tensor([[0, 1, 3, 4, 5], [1, 2, 4, 5, 6]])
    return nodes, queries, anchors, batch, edges


def test_all_screen_models_produce_one_score_per_candidate():
    inputs = _inputs()
    for name in SCREEN_MODELS:
        model = build_operator_model(name, 8, 4, layers=1, offset_directions=4, dropout=0.0)
        scores = model(*inputs[:-1], inputs[-1] if model.uses_topology else None)
        assert scores.shape == (7,)
        counts = model_parameter_counts(model)
        assert counts["parameters"] == counts["trainable_parameters"] > 0


def test_non_graph_operators_ignore_topology_and_k_is_fixed():
    nodes, queries, anchors, batch, edges = _inputs()
    empty = torch.empty((2, 0), dtype=torch.long)
    for name in ("plain_mlp", "offset_mlp", "offset_mlp_k4"):
        model = build_operator_model(name, 8, 4, dropout=0.0).eval()
        assert torch.equal(
            model(nodes, queries, anchors, batch, edges),
            model(nodes, queries, anchors, batch, empty),
        )
    k_model = build_operator_model("offset_mlp_k4", 8, 4)
    assert k_model.directions == 4


def test_message_passing_requires_topology():
    nodes, queries, anchors, batch, _edges = _inputs()
    model = build_operator_model("gcn", 8, 4)
    try:
        model(nodes, queries, anchors, batch, None)
    except ValueError as exc:
        assert "edge_index" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("GCN accepted a missing topology")


def test_coverage_offset_exposes_k_direction_scores_without_topology():
    nodes, queries, anchors, batch, edges = _inputs()
    model = build_operator_model(
        COVERAGE_OFFSET_MODEL,
        8,
        4,
        offset_directions=4,
        dropout=0.0,
    ).eval()
    directional, targets = model.directional_scores(nodes, queries, anchors, batch, edges)
    assert directional.shape == (7, 4)
    assert targets.shape == (2, 4, 4)
    assert torch.equal(model(nodes, queries, anchors, batch, edges), directional.max(dim=1).values)
    assert model.uses_topology is False


def test_explicit_feature_models_are_parameter_matched_and_have_no_message_passing():
    nodes, queries, _anchors, batch, _edges = _inputs()
    target = 1_000
    for name in STRUCTURE_AWARE_MODELS:
        model = build_explicit_feature_mlp(
            name,
            embedding_dim=8,
            projection_dim=4,
            static_dim=7,
            local_dim=10,
            target_parameters=target,
            dropout=0.0,
            temperature=0.07,
        ).eval()
        structural = None
        if model.structural_dim:
            structural = torch.randn(nodes.shape[0], model.structural_dim)
        scores = model.forward_explicit(nodes, queries, batch, structural)
        assert scores.shape == (nodes.shape[0],)
        assert model.uses_topology is False
        assert not any("conv" in type(module).__name__.lower() for module in model.modules())
        assert abs(model_parameter_counts(model)["parameters"] - target) < 64
