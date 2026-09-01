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


def test_online_systems_lands_as_one_file_per_dataset() -> None:
    # Package D writes one result per dataset, and analyze_online_systems.py
    # reads outputs/online_systems/<dataset>.json rather than a condition
    # directory.
    assert _local_relative(
        "online_systems", "webqsp", "", "result.json"
    ) == Path("webqsp.json")


def test_online_systems_is_discovered_at_its_own_remote_depth() -> None:
    # Its remote path is dataset/fingerprint/result.json with no condition
    # level. Discovery filtered on a fixed depth of four, so without this the
    # package would download nothing and report zero conditions rather than
    # failing.
    assert PACKAGES["online_systems"].remote_depth == 3
    assert PACKAGES["candidate_budget"].remote_depth == 4
    assert PACKAGES["online_systems"].complete_status == (
        "UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_COMPLETE"
    )


def test_phase_confirmation_cells_keep_both_files_together() -> None:
    # The confirmation writes packed per-query metrics beside its result, and
    # the paired statistics need both. A flat file per cell, as the screen uses,
    # would have nowhere to put the second artifact.
    assert _local_relative(
        "phase_confirmation", "webqsp", "degree_rewire_0p25", "result.json"
    ) == Path("webqsp/degree_rewire/rate_0p25/result.json")
    assert _local_relative(
        "phase_confirmation", "webqsp", "degree_rewire_0p25", "query_metrics.npz"
    ) == Path("webqsp/degree_rewire/rate_0p25/query_metrics.npz")


def test_an_unparsable_phase_confirmation_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="phase-confirmation condition"):
        _local_relative("phase_confirmation", "webqsp", "noseparator", "result.json")
