from scripts.analyze_online_systems import _ratio


def test_online_ratio_is_directionally_explicit() -> None:
    assert _ratio(2.0, 4.0) == 0.5
