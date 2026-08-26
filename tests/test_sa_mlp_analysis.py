from scripts.analyze_sa_mlp_screen import _pct


def test_screen_percentage_format_is_paper_percentage_points() -> None:
    assert _pct(0.300407) == "30.04"
