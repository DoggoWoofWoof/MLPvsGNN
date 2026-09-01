import pytest

from scripts.compile_package_c import REPO_ROOT, attainment, decompose_budget_step


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


def test_the_report_shows_every_jointly_required_column() -> None:
    # The protocol requires effectiveness, pool ceilings, induced topology and
    # cached cost to be readable side by side. A number reported without its
    # ceiling invites exactly the misreading this package exists to prevent,
    # so dropping a column here is a scientific regression, not a layout change.
    markdown = (REPO_ROOT / "docs" / "CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md").read_text(
        encoding="utf-8"
    )
    required = [
        "Coverage",
        "GoldFrac",
        "AnyGold",
        "AllGold",
        "Ceil@1",
        "Ceil@5",
        "Ceil@20",
        "QLS R@1",
        "QLS R@5",
        "QLS R@20",
        "GNN R@1",
        "GNN R@5",
        "GNN R@20",
        "Ceil-QLS",
        "Ceil-GNN",
        "QLS MRR",
        "GNN MRR",
        "FullCov ceil",
        "Nodes/q",
        "Edges/q",
        "Density",
        "Components",
        "QLS ms/q",
        "GNN ms/q",
        "from ceiling",
        "from ranking",
    ]
    missing = [column for column in required if column not in markdown]
    assert not missing, f"joint report no longer shows: {missing}"


def test_the_report_states_that_its_latency_is_cached_only() -> None:
    # Cached operator latency, uncached post-retrieval latency and raw-query
    # end-to-end latency are three different quantities. The report carries the
    # first, so it must say so where the numbers are.
    markdown = (REPO_ROOT / "docs" / "CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md").read_text(
        encoding="utf-8"
    )
    assert "cached-operator latency only" in markdown
    assert "not uncached post-retrieval latency" in markdown
