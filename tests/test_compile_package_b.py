"""Package B may only be read against the ceiling that actually describes its pool.

Attaching a ceiling measured on a different candidate pool would silently
rescale every gap in the report, so the join is contract-checked rather than
assumed.
"""

import json
from pathlib import Path

import pytest

from scripts.compile_package_b import (
    RELATIONAL_FAMILY,
    SIMILARITY_FAMILY,
    attainment,
    compile_joint,
)

CONTRACT = "a" * 64
OTHER_CONTRACT = "b" * 64
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


def _analysis_row(dataset: str, family: str, recall5: float) -> dict:
    models = {
        model: {metric: {"mean": recall5} for metric in METRICS}
        for model in ("sa_mlp", "seed_aware_gnn")
    }
    return {
        "dataset": dataset,
        "family": family,
        "directed_edges": 1000,
        "models": models,
        "seed_aware_gnn_minus_sa_mlp": {"recall@5": {"seed_effect": {"mean": 0.0}}},
    }


@pytest.fixture
def package_b(tmp_path: Path) -> tuple[dict, Path, Path]:
    root = tmp_path / "edge_provenance"
    headroom_root = tmp_path / "candidate_headroom"
    headroom_root.mkdir(parents=True)

    for family in (RELATIONAL_FAMILY, SIMILARITY_FAMILY):
        cell = root / "webqsp" / family
        cell.mkdir(parents=True)
        (cell / "result.json").write_text(
            json.dumps({"candidate_contract": {"observed_contract_sha256": CONTRACT}}),
            encoding="utf-8",
        )

    (headroom_root / "webqsp.json").write_text(
        json.dumps(
            {
                "status": "CANDIDATE_HEADROOM_DIAGNOSTIC_COMPLETE",
                "candidate_contract": {"observed_contract_sha256": CONTRACT},
                "headroom": {
                    "test": {
                        "frozen_union": {
                            "coverage_micro": 0.5,
                            "gold_fraction_at_pool_macro": 0.5,
                            "any_gold_at_pool": 0.6,
                            "all_gold_at_pool": 0.4,
                            "recall_ceiling@1": 0.3,
                            "recall_ceiling@5": 0.5,
                            "recall_ceiling@20": 0.5,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = {
        "status": "EDGE_PROVENANCE_ALL_DATASETS_ANALYZED",
        "datasets": [
            _analysis_row("webqsp", RELATIONAL_FAMILY, 0.40),
            _analysis_row("webqsp", SIMILARITY_FAMILY, 0.25),
        ],
    }
    return analysis, root, headroom_root


def test_attainment_is_the_fraction_of_the_shared_ceiling_that_was_reached() -> None:
    assert attainment(0.40, 0.50) == pytest.approx(0.80)


def test_attainment_is_undefined_rather_than_zero_when_nothing_is_reachable() -> None:
    # Dividing by an empty pool would report a ranking failure that no ranker
    # could have avoided.
    assert attainment(0.0, 0.0) is None


def test_the_shared_ceiling_is_attached_to_every_family(package_b) -> None:
    analysis, root, headroom_root = package_b
    joint = compile_joint(analysis, root, headroom_root)
    ceilings = {row["candidate_pool"]["recall_ceiling@5"] for row in joint["rows"]}
    assert ceilings == {0.5}, "families of one dataset must share one ceiling"


def test_family_differences_survive_as_attainment_differences(package_b) -> None:
    # The ceiling cancels between families, so a difference in attainment is a
    # topology effect and not an upstream candidate effect.
    analysis, root, headroom_root = package_b
    joint = compile_joint(analysis, root, headroom_root)
    by_family = {row["family"]: row for row in joint["rows"]}
    assert by_family[RELATIONAL_FAMILY]["ceiling_gaps"]["sa_mlp"]["attainment@5"] == pytest.approx(
        0.80
    )
    assert by_family[SIMILARITY_FAMILY]["ceiling_gaps"]["sa_mlp"]["attainment@5"] == pytest.approx(
        0.50
    )


def test_relational_and_similarity_families_are_reported_as_a_pair(package_b) -> None:
    # The provenance question needs both halves side by side; reporting either
    # alone cannot answer whether message passing used relations or embeddings.
    analysis, root, headroom_root = package_b
    joint = compile_joint(analysis, root, headroom_root)
    assert len(joint["provenance_contrast"]) == 1
    contrast = joint["provenance_contrast"][0]["models"]["sa_mlp"]["recall@5"]
    assert contrast["relational_minus_similarity"] == pytest.approx(0.15)


def test_a_ceiling_measured_on_a_different_pool_is_refused(package_b) -> None:
    # The failure this file exists to prevent: a ceiling that does not describe
    # the pool these models were trained against would rescale every gap.
    analysis, root, headroom_root = package_b
    payload = json.loads((headroom_root / "webqsp.json").read_text(encoding="utf-8"))
    payload["candidate_contract"]["observed_contract_sha256"] = OTHER_CONTRACT
    (headroom_root / "webqsp.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="would not describe these results"):
        compile_joint(analysis, root, headroom_root)


def test_an_incomplete_headroom_diagnostic_is_refused(package_b) -> None:
    analysis, root, headroom_root = package_b
    payload = json.loads((headroom_root / "webqsp.json").read_text(encoding="utf-8"))
    payload["status"] = "CANDIDATE_HEADROOM_IN_PROGRESS"
    (headroom_root / "webqsp.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        compile_joint(analysis, root, headroom_root)


def test_families_that_disagree_on_the_candidate_contract_are_refused(package_b) -> None:
    # Package B's whole design is that only the edge family changes. Families
    # on different candidate pools would not be comparable at all.
    analysis, root, headroom_root = package_b
    stray = root / "webqsp" / SIMILARITY_FAMILY / "result.json"
    stray.write_text(
        json.dumps({"candidate_contract": {"observed_contract_sha256": OTHER_CONTRACT}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagree on the candidate contract"):
        compile_joint(analysis, root, headroom_root)


def test_a_partial_analysis_is_refused_rather_than_compiled(package_b) -> None:
    # Compiling before every dataset is analyzed would freeze a report whose
    # missing rows look like absent effects rather than absent runs.
    analysis, root, headroom_root = package_b
    analysis["status"] = "EDGE_PROVENANCE_DATASET_FAMILY_COMPLETE"
    with pytest.raises(ValueError, match="Refusing to compile"):
        compile_joint(analysis, root, headroom_root)
