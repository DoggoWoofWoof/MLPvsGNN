from pathlib import Path

import pytest

from scripts.fetch_modal_results import PACKAGES, _local_relative


def test_budget_conditions_keep_their_remote_shape() -> None:
    assert _local_relative(
        "candidate_budget", "squad_clean", "budget_400", "result.json"
    ) == Path("squad_clean/budget_400/result.json")


def test_provenance_conditions_keep_their_remote_shape() -> None:
    assert _local_relative(
        "edge_provenance", "webqsp", "knn_only", "query_metrics.npz"
    ) == Path("webqsp/knn_only/query_metrics.npz")


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("degree_rewire_0p10", "2wiki_clean/degree_rewire/rate_0p10.json"),
        ("hub_injection_1p00", "2wiki_clean/hub_injection/rate_1p00.json"),
        ("feature_mask_0p25", "2wiki_clean/feature_mask/rate_0p25.json"),
        ("random_add_0p50", "2wiki_clean/random_add/rate_0p50.json"),
    ],
)
def test_phase_screen_condition_is_split_into_axis_and_rate(
    condition: str, expected: str
) -> None:
    # Remote stores one directory per axis-and-rate; the analyzer reads a
    # directory per axis with one file per rate. Axis names contain
    # underscores, so only the final segment is the rate.
    assert _local_relative("phase_screen", "2wiki_clean", condition, "result.json") == (
        Path(expected)
    )


def test_an_unparsable_phase_screen_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unparsable"):
        _local_relative("phase_screen", "2wiki_clean", "noseparator", "result.json")


def test_every_package_declares_the_status_its_analyzer_requires() -> None:
    assert PACKAGES["candidate_budget"].complete_status == (
        "CANDIDATE_BUDGET_DATASET_COMPLETE"
    )
    assert PACKAGES["edge_provenance"].complete_status == (
        "EDGE_PROVENANCE_DATASET_FAMILY_COMPLETE"
    )
    assert PACKAGES["phase_screen"].complete_status == (
        "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE"
    )
