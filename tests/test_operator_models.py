import torch

from mp_retrieval.operator_models import (
    COVERAGE_OFFSET_MODEL,
    SCREEN_MODELS,
    SEED_ONLY_MODEL,
    STRUCTURE_AWARE_MODELS,
    build_explicit_feature_mlp,
    build_operator_model,
    build_seed_aware_message_passing,
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


def test_seed_only_control_accepts_exactly_one_fixed_indicator():
    nodes, queries, _anchors, batch, _edges = _inputs()
    model = build_explicit_feature_mlp(
        SEED_ONLY_MODEL,
        embedding_dim=8,
        projection_dim=4,
        static_dim=0,
        local_dim=1,
        target_parameters=1_000,
        dropout=0.0,
        temperature=0.07,
    ).eval()
    scores = model.forward_explicit(
        nodes,
        queries,
        batch,
        torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]),
    )
    assert scores.shape == (7,)
    assert model.include_interactions is True
    assert model.include_static is False
    assert model.local_dim == 1


def test_seed_aware_gnn_adds_only_one_input_weight_per_hidden_channel():
    nodes, queries, _anchors, batch, edges = _inputs()
    baseline = build_operator_model("gcn", 8, 4, layers=1, dropout=0.0)
    control = build_seed_aware_message_passing("gcn", 8, 4, layers=1, dropout=0.0)
    seed = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    scores = control.forward_seed_aware(nodes, queries, batch, edges, seed)
    assert scores.shape == (7,)
    assert (
        model_parameter_counts(control)["parameters"]
        - model_parameter_counts(baseline)["parameters"]
        == 4
    )
