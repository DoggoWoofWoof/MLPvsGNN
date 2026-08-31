from scripts.analyze_phase_screen import select_confirmation_rates


def test_phase_selection_no_crossing_keeps_only_endpoints() -> None:
    points = [(0.0, 0.1), (0.1, 0.08), (0.25, 0.05), (1.0, 0.01)]
    assert select_confirmation_rates(points) == [0.0, 1.0]


def test_phase_selection_keeps_each_crossing_bracket() -> None:
    points = [(0.0, 0.1), (0.1, 0.02), (0.25, -0.01), (0.5, -0.03), (1.0, 0.04)]
    assert select_confirmation_rates(points) == [0.0, 0.1, 0.25, 0.5, 1.0]


def test_phase_selection_exact_zero_keeps_neighbors_and_endpoints() -> None:
    points = [(0.0, 0.1), (0.1, 0.02), (0.25, 0.0), (0.5, -0.03), (1.0, -0.04)]
    assert select_confirmation_rates(points) == [0.0, 0.1, 0.25, 0.5, 1.0]
