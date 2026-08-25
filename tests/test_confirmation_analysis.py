import json
from pathlib import Path

import pytest

from scripts.analyze_confirmation import _paired_group_gap, _query_hops, mean_sd_ci


def _metric_row(value: float) -> dict[str, float]:
    return {
        "recall@1": value,
        "recall@5": value,
        "recall@20": value,
        "mrr": value,
        "full_coverage@20": value,
    }


def _per_query_model(q2: float, q4: float) -> dict:
    return {
        "seeds": {
            str(seed): {
                "per_query": {
                    "q2": _metric_row(q2 + seed * 0.001),
                    "q4": _metric_row(q4 + seed * 0.001),
                }
            }
            for seed in range(5)
        }
    }


def test_answer_count_gap_reports_monotonic_coverage_loss():
    left = _per_query_model(0.9, 0.4)
    right = _per_query_model(0.8, 0.8)
    gaps = _paired_group_gap(left, right, {"q2": 2, "q4": 4})
    assert gaps["2"]["recall@20"]["mean"] == pytest.approx(0.1)
    assert gaps["4"]["recall@20"]["mean"] == pytest.approx(-0.4)
    assert gaps["trend"]["monotonic_nonincreasing"] is True


def test_musique_hops_are_recovered_without_using_gold_count(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ids": ["musique_clean_q_train_0", "musique_clean_q_train_1"],
                "split_indices": {"test": [0, 1]},
            }
        ),
        encoding="utf-8",
    )
    raw = tmp_path / "musique.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "train_0",
                        "metadata": {"question_decomposition": [{}, {}]},
                    }
                ),
                json.dumps(
                    {
                        "id": "train_1",
                        "metadata": {"question_decomposition": [{}, {}, {}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    hops, provenance = _query_hops({"dataset": "musique_clean"}, manifest, raw)
    assert hops == {"musique_clean_q_train_0": 2, "musique_clean_q_train_1": 3}
    assert provenance["status"] == "complete"


def test_five_seed_interval_uses_sample_variance():
    interval = mean_sd_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert interval["n"] == 5
    assert interval["mean"] == pytest.approx(0.3)
    assert interval["ci95_low"] < 0.3 < interval["ci95_high"]
