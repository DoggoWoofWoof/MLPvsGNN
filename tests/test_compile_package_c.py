import pytest

from scripts.compile_package_c import attainment, decompose_budget_step


def test_attainment_is_the_fraction_of_the_ceiling_that_was_reached() -> None:
    assert attainment(0.30, 0.40) == pytest.approx(0.75)


def test_attainment_is_undefined_rather_than_zero_when_no_gold_is_reachable() -> None:
    # A zero ceiling means no ranking could score. A ratio would be an artefact.
    assert attainment(0.0, 0.0) is None


def test_budget_step_terms_sum_exactly_to_the_observed_change() -> None:
    step = decompose_budget_step(
        lower_recall=0.30, lower_ceiling=0.40, upper_recall=0.45, upper_ceiling=0.60
    )
    assert step["attributable"] is True
    assert step["observed_recall_change"] == pytest.approx(0.15)
    assert step["ceiling_effect"] + step["ranking_effect"] == pytest.approx(
        step["observed_recall_change"]
    )


def test_a_pure_ceiling_rise_is_not_attributed_to_ranking() -> None:
    # Attainment is 0.75 at both budgets, so nothing was gained by reranking.
    step = decompose_budget_step(
        lower_recall=0.30, lower_ceiling=0.40, upper_recall=0.60, upper_ceiling=0.80
    )
    assert step["ranking_effect"] == pytest.approx(0.0)
    assert step["ceiling_effect"] == pytest.approx(0.30)


def test_a_pure_ranking_gain_is_not_attributed_to_the_ceiling() -> None:
    # The ceiling is unchanged, so the whole gain is reranking.
    step = decompose_budget_step(
        lower_recall=0.30, lower_ceiling=0.60, upper_recall=0.45, upper_ceiling=0.60
    )
    assert step["ceiling_effect"] == pytest.approx(0.0)
    assert step["ranking_effect"] == pytest.approx(0.15)


def test_a_rising_ceiling_with_falling_attainment_splits_with_opposite_signs() -> None:
    # Raw recall improves, but the ranker used less of what it was given. The
    # decomposition must show that rather than crediting the ranker.
    step = decompose_budget_step(
        lower_recall=0.30, lower_ceiling=0.40, upper_recall=0.35, upper_ceiling=0.70
    )
    assert step["observed_recall_change"] == pytest.approx(0.05)
    assert step["ceiling_effect"] > 0.0
    assert step["ranking_effect"] < 0.0
    assert step["ceiling_effect"] + step["ranking_effect"] == pytest.approx(0.05)


def test_an_unreachable_pool_is_reported_as_unattributable_not_as_zero() -> None:
    step = decompose_budget_step(
        lower_recall=0.0, lower_ceiling=0.0, upper_recall=0.10, upper_ceiling=0.20
    )
    assert step["attributable"] is False
    assert step["ceiling_effect"] is None
    assert step["ranking_effect"] is None
    assert step["observed_recall_change"] == pytest.approx(0.10)
